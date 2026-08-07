#!/usr/bin/env python3
"""
make_synthetic_layer.py — materialise a layer dir whose "features" contain no model.

WHY THIS EXISTS
---------------
Every metric in this project reads a layer directory: Z.npy plus uids.json,
lengths.npy, offsets.npy, sequences.json. cpu_stage.py, experiment_interplm_metric.py
and experiment_concept_f1.py all take --layer-dir and nothing else.

So the cheapest way to ask "can this metric be satisfied without a model?" is not
to modify each metric -- it is to hand each metric a layer directory in which the
features are indicator functions of residue identity. Nothing downstream needs to
know. The metric code stays byte-identical to the version that produced the
paper's numbers, which is the point: a result obtained by editing the estimator
is a result about the edit.

The features are the same vocabulary as experiment_synthetic_composition.py:
20 single-amino-acid indicators, 8 chemical-class indicators, and one all-ones
column that is the trivial floor for any precision/recall metric.

Feature index -> name is written to feature_names.json so downstream CSVs
(which are indexed by integer feature id) can be read back.

USAGE
  python make_synthetic_layer.py --src outputs_ctrl/ctrl_mlm_A/layer_6 \
                                 --out outputs_synthetic/composition

  # then, unmodified:
  python experiment_interplm_metric.py --layer-dir outputs_synthetic/composition \
      --out results_stress/interplm_synthetic.csv
  python cpu_stage.py --layer-dir outputs_synthetic/composition --n-shuffles 5 ...

--src supplies only the residue ordering and sequences. No activation is read,
so the output is a property of the evaluation set, identical for every model.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"

CLASSES = {
    "hydrophobic": set("AVLIMFWC"),
    "charged":     set("DEKR"),
    "polar":       set("STNQYH"),
    "aromatic":    set("FWYH"),
    "small":       set("AGSTCVPND"),
    "tiny":        set("AGS"),
    "negative":    set("DE"),
    "positive":    set("KR"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="layer dir supplying uids/lengths/offsets/sequences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--no-trivial", action="store_true",
                    help="omit the all-ones column (it saturates any top-k gate)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="uniform noise in [0,jitter) added to active entries. "
                         "REQUIRED for any argmax-anchored estimator: a binary "
                         "feature ties everywhere, np.argmax returns the FIRST "
                         "index, and the anchor is pinned to the N-terminus "
                         "(measured mean relative position 0.009 for "
                         "class:hydrophobic against ~0.5 for a real SAE feature). "
                         "That depresses structural scores for reasons that have "
                         "nothing to do with composition. 1e-3 breaks ties "
                         "uniformly while leaving any gate above 0.6 unaffected.")
    ap.add_argument("--jitter-seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    uids = [str(u) for u in json.loads((src / "uids.json").read_text())]
    lengths = np.load(src / "lengths.npy")
    seqs_raw = json.loads((src / "sequences.json").read_text())
    seqs = seqs_raw if isinstance(seqs_raw, dict) else dict(zip(uids, seqs_raw))

    n_res = int(np.sum(lengths))
    codes = np.empty(n_res, dtype="<U1")
    off = 0
    for u, L in zip(uids, lengths):
        L = int(L)
        s = str(seqs[u])[:L]
        if len(s) < L:
            raise SystemExit(f"  {u}: sequence {len(s)} < lengths entry {L}; "
                             f"padding would misalign every downstream row.")
        codes[off:off + L] = list(s)
        off += L

    names, cols = [], []
    for a in AA:
        names.append(f"aa:{a}")
        cols.append((codes == a))
    for cname, members in CLASSES.items():
        names.append(f"class:{cname}")
        cols.append(np.isin(codes, list(members)))
    if not args.no_trivial:
        names.append("TRIVIAL:all-ones")
        cols.append(np.ones(n_res, dtype=bool))

    Z = np.stack(cols, axis=1).astype(args.dtype)

    if args.jitter > 0:
        # Break argmax ties uniformly at random among a feature's active
        # residues. Applied only where the indicator is 1, so the SET of active
        # residues is unchanged and any threshold gate sees the same members.
        rng = np.random.RandomState(args.jitter_seed)
        noise = rng.uniform(0.0, args.jitter, size=Z.shape).astype(Z.dtype)
        Z = np.where(Z > 0, Z + noise, Z).astype(args.dtype)
        print(f"  jitter {args.jitter:g} applied to active entries "
              f"(seed {args.jitter_seed}) — anchors are now uniform over instances")

    np.save(out / "Z.npy", Z)

    # offsets: recompute rather than copy, so a mismatched src cannot propagate
    offsets = np.concatenate([[0], np.cumsum(lengths.astype(np.int64))[:-1]])
    np.save(out / "offsets.npy", offsets)
    np.save(out / "lengths.npy", lengths)
    for f in ("uids.json", "sequences.json"):
        shutil.copy(src / f, out / f)
    (out / "feature_names.json").write_text(json.dumps(names, indent=1))

    occ = Z.astype(np.float32).mean(axis=0)
    (out / "META.json").write_text(json.dumps({
        "model": "SYNTHETIC-composition-indicators",
        "note": "no model, no SAE; features are indicator functions of residue identity",
        "src_layer_dir": str(src),
        "embed_dim": 0,
        "sae_hidden_dim": int(Z.shape[1]),
        "n_residues": n_res,
        "feature_names": names,
        "occupancy": {n: float(o) for n, o in zip(names, occ)},
    }, indent=1))

    print(f"  wrote {out}/Z.npy  {Z.shape} {Z.dtype}")
    print(f"  {len(names)} features, {n_res} residues, {len(uids)} proteins")
    print("  occupancy range %.4f - %.4f" % (occ.min(), occ.max()))
    print()
    print("  NOTE: a feature whose occupancy exceeds a metric's top-k fraction is")
    print("  silently zeroed by percentile gating (see cpu_stage._cohens_d_vectorized).")
    print("  The all-ones column has occupancy 1.0 and WILL be zeroed by any such")
    print("  gate -- it is meaningful only for precision/recall metrics like")
    print("  concept-F1. Use --no-trivial when feeding cpu_stage.")


if __name__ == "__main__":
    main()
