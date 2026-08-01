#!/usr/bin/env python3
"""
experiment_synthetic_composition.py — does L_struct reward composition alone?

WHY THIS EXISTS
---------------
Section 5.4 rejects "composition clustering" using an amino-acid SELECTIVITY measure:
one minus the normalised entropy of a feature's activation over the 20 individual
types. That measure is blind to a CLASS detector. A feature firing on all
hydrophobic residues is spread over ~8 types and therefore scores as maximally
UNSELECTIVE, so the existing test cannot see it.

This matters because the permutation null shuffles residue positions against a FIXED
structure. Any property with spatial autocorrelation in the fold therefore produces
excess structural co-activation without encoding anything about structure.
Hydrophobicity is the obvious case: hydrophobic residues are disproportionately
buried, buried residues have high contact degree, and their long-range partners are
disproportionately hydrophobic too.

THE TEST
--------
Build synthetic "features" that are pure indicator functions of residue identity --
no model, no SAE, no training. Push them through the IDENTICAL metric path used for
every real number in this project (same graphs, same permutation null, same
topk_frac, same struct_delta = observed - shuffled).

  * 20 single-type indicators (one per amino acid)
  * class indicators: hydrophobic, charged, polar, aromatic, small, tiny

If a class indicator -- hydrophobic above all -- scores an L_struct comparable to
real SAE features, then L_struct rewards spatial autocorrelation of composition and
the composition account is NOT refuted. That would close the open question in the
mechanism section and give a more general result than the one currently claimed.

Reference points printed alongside: the real per-feature L_struct distribution from
the same cell, so the synthetic scores are directly comparable.

USAGE
  python experiment_synthetic_composition.py --layer-dir outputs_ctrl/ctrl_mlm_A/layer_6
  python experiment_synthetic_composition.py --layer-dir ... --cutoff 8 --seq-gap 12

NOTE: --n-shuffles defaults to 5 here, matching every other number in the project.
The cpu_stage default of 3 computes a different statistic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from cpu_stage import (
    load_ref_seqs, build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
    _process_struct_seq_chunk_v3,
)

AA = "ACDEFGHIKLMNPQRSTVWY"

# Standard groupings. Hydrophobic is the one the argument turns on.
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


def residue_codes(layer_dir: Path):
    """Per-row amino-acid character, aligned to the Z row order."""
    seqs = json.loads((layer_dir / "sequences.json").read_text())
    uids = json.loads((layer_dir / "uids.json").read_text())
    lengths = np.load(layer_dir / "lengths.npy")
    ordered = [seqs[str(u)] for u in uids] if isinstance(seqs, dict) else list(seqs)
    out = np.empty(int(np.sum(lengths)), dtype="<U1")
    off = 0
    for s, L in zip(ordered, lengths):
        L = int(L)
        chars = list(str(s)[:L])
        if len(chars) < L:
            raise SystemExit(f"  sequence shorter ({len(chars)}) than its lengths entry "
                             f"({L}); padding this would misalign every downstream row.")
        out[off:off + L] = chars
        off += L
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--seq-gap", type=int, default=12)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--topk-frac", type=float, default=0.10)
    ap.add_argument("--out", default="results_synthetic_composition.csv")
    ap.add_argument("--compare-real", action="store_true",
                    help="rescore the cell's REAL features at this same --topk-frac so "
                         "both sides are scored under one rule. Without it the printed "
                         "real percentiles come from struct_seq_metrics.csv, which was "
                         "computed at the project default of 0.10.")
    args = ap.parse_args()

    layer_dir = Path(args.layer_dir)
    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.is_dir() or not any(pdb_dir.iterdir()):
        raise SystemExit(f"  --pdb-dir {pdb_dir} missing or empty. Without it every "
                         f"pdb_path.exists() is False and the contact graph comes back "
                         f"empty, which looks like a result rather than an error.")
    codes = residue_codes(layer_dir)
    uids = [str(u) for u in json.loads((layer_dir / "uids.json").read_text())]
    lengths = np.load(layer_dir / "lengths.npy")
    ref_seqs = load_ref_seqs(layer_dir)
    n_res = len(codes)

    print(f"  cell        : {layer_dir}")
    print(f"  residues    : {n_res:,}   proteins: {len(uids):,}")
    print(f"  relation    : Ca < {args.cutoff} A, |i-j| >= {args.seq_gap}, "
          f"{args.n_shuffles} shuffles")

    # identical graphs to every other number in the project
    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, lengths, ref_seqs, pdb_dir, args.n_jobs,
        contact_cutoff=args.cutoff, seq_gap_min=args.seq_gap)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)
    del seq_adj, struct_adj
    print(f"  contact edges: {A_struct.nnz:,}")

    # synthetic feature matrix: pure indicators, no model anywhere
    names, cols = [], []
    for a in AA:
        names.append(f"aa:{a}"); cols.append((codes == a).astype(np.float16))
    for cname, members in CLASSES.items():
        names.append(f"class:{cname}")
        cols.append(np.isin(codes, list(members)).astype(np.float16))
    Z = np.stack(cols, axis=1)
    frac = Z.mean(axis=0)
    print(f"  synthetic features: {Z.shape[1]} "
          f"(occupancy {frac.min():.3f}-{frac.max():.3f})")

    # A BINARY feature whose occupancy exceeds topk_frac is silently zeroed:
    # thresh = percentile(acts, 100*(1-topk_frac)) is exactly 1.0, `acts > thresh`
    # selects nothing, n_active = 0, and the kernel forces d = 0.0. Refuse to emit
    # that rather than let a zero be read as a null result. NB this degeneracy needs
    # values TIED at the threshold, so it affects binary indicators only -- real
    # continuous SAE features are not at risk.
    over = [(n, f) for n, f in zip(names, frac) if f >= args.topk_frac]
    if over:
        print(f"\n  !! {len(over)} feature(s) have occupancy >= topk_frac="
              f"{args.topk_frac}; these would be silently zeroed:")
        for n, f in over:
            print(f"       {n:22} occupancy {f:.4f}")
        raise SystemExit(
            f"  refusing to emit zeros. Re-run with --topk-frac above "
            f"{max(f for _, f in over):.3f}, or drop those features.")
    margin = min(args.topk_frac - f for f in frac)
    tight = [n for n, f in zip(names, frac) if args.topk_frac - f < 0.01]
    if tight:
        print(f"  !! within 0.01 of the threshold (would vanish on another "
              f"dataset): {', '.join(tight)}")
    print(f"  smallest margin to topk_frac: {margin:.4f}\n")

    perms = build_protein_permutations(lengths, args.n_shuffles)
    n_features = Z.shape[1]
    chunk = 64
    n_chunks = (n_features + chunk - 1) // chunk
    res = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk, Z, None, A_seq, deg_seq, A_struct, deg_struct,
            perms, n_features, args.topk_frac)
        for ci in range(n_chunks))
    idx = np.concatenate([r[0] for r in res])
    obs = np.concatenate([r[2] for r in res])
    sh = np.concatenate([r[4] for r in res])
    order = np.argsort(idx)
    struct_delta = (obs - sh)[order]

    import pandas as pd
    df = pd.DataFrame(dict(feature=names, occupancy=frac,
                           struct_delta=struct_delta)).sort_values(
        "struct_delta", ascending=False)
    df.to_csv(args.out, index=False)

    # reference: the real SAE features from this very cell
    ref = layer_dir / "struct_seq_metrics.csv"
    print("  SYNTHETIC FEATURES, ranked by L_struct")
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    real, how = None, ""
    if args.compare_real and (layer_dir / "Z.npy").exists():
        print("\n  rescoring the REAL features at this same topk_frac so both "
              "sides are scored under one rule...")
        Zr = np.load(layer_dir / "Z.npy", mmap_mode="r")
        nf = int(Zr.shape[1]); ch = 256
        rr = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(_process_struct_seq_chunk_v3)(
                ci, ch, Zr, None, A_seq, deg_seq, A_struct, deg_struct,
                perms, nf, args.topk_frac)
            for ci in range((nf + ch - 1) // ch))
        ri = np.concatenate([r[0] for r in rr])
        real = ((np.concatenate([r[2] for r in rr])
                 - np.concatenate([r[4] for r in rr]))[np.argsort(ri)])
        np.save(args.out.replace(".csv", "_realmatched.npy"), real)
        how = f"rescored at topk_frac={args.topk_frac}, like-for-like"
    elif ref.exists():
        real = pd.read_csv(ref)["struct_delta"].to_numpy(float)
        how = ("from struct_seq_metrics.csv, computed at the project default "
               "topk_frac=0.10 — pass --compare-real for like-for-like")

    if real is not None:
        print(f"\n  REAL SAE features in the same cell (n={len(real):,})")
        print(f"    [{how}]")
        for q in (50, 75, 90, 99):
            print(f"    p{q:<3} {np.percentile(real, q):+.4f}")
        print(f"    max  {real.max():+.4f}    mean {real.mean():+.4f}")
        # Report the class rows separately: their occupancy is in the same range
        # as a real feature's top-decile active set, so they are the fair
        # comparison. A single-type indicator selects far fewer residues.
        for label, sub in (("best overall", df),
                           ("best CLASS  ", df[df.feature.str.startswith("class:")])):
            if not len(sub):
                continue
            best = sub.struct_delta.max()
            row = sub.loc[sub.struct_delta.idxmax()]
            pct = (real < best).mean() * 100
            print(f"  >> {label} ({row.feature}, occupancy {row.occupancy:.3f}) "
                  f"= {best:+.4f}, beating {pct:.1f}% of real learned features")
    else:
        print("\n  [no real-feature reference found in this cell]")
    print(f"\n  written: {args.out}")


if __name__ == "__main__":
    main()
