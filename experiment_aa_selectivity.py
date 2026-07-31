#!/usr/bin/env python3
"""
experiment_aa_selectivity.py — is the co-activation metric detecting amino-acid composition?

THE CLAIM UNDER TEST (the paper's central explanatory claim, §4.5)
  Chemically similar residues pack together in folded proteins, so a feature that detects
  amino-acid class will appear to detect spatial proximity under a co-activation metric. If
  that is what `L_struct` is picking up, then features that are more amino-acid-selective
  should have systematically higher `struct_delta`.

  PREDICTION: Spearman(AA selectivity, struct_delta) > 0, and materially so.

  FALSIFIED IF: the correlation is ~0. Then composition clustering is not the explanation and
  §4.5 must be withdrawn or replaced — the shuffled-sequence result would still stand as a
  fact, but we would have no account of it.

SELECTIVITY MEASURE
  For feature j, let m_ja be the total activation mass of j on residues of amino-acid type a,
  normalised to a distribution p_ja over the 20 types. Selectivity is the normalised negative
  entropy

      S_j = 1 - H(p_j) / log(20)

  so S_j = 0 for a feature spread evenly across all amino acids and S_j -> 1 for a feature
  firing on exactly one type. This is background-corrected by dividing by each type's residue
  frequency before normalising, so an abundant residue does not look selective by virtue of
  being common.

  A second, cruder measure is reported alongside: `top1_share`, the fraction of (background
  corrected) mass on the single most-preferred type. If the two disagree the result is fragile
  and should not be reported.

CONTROL
  The same correlation against `seq_delta` (sequential-neighbour locality). Amino-acid identity
  is spatially clustered but also, weakly, sequentially clustered, so a composition detector
  should show up in both. What would be surprising — and would weaken the explanation — is a
  strong structural correlation with no sequential one.

NOTE ON SCOPE
  Runs on whichever cells have Z.npy on disk. In this repository that is the ESM-2 / RITA pair,
  not the controlled models, whose Z was pruned. Testing the mechanism on real production models
  is arguably the stronger test anyway, but it is a different model pair from the one the
  disagreement was measured on, and the write-up must say so.

Usage
  .venv/bin/python experiment_aa_selectivity.py --root outputs_outlier \\
      --cells esm2_ptn:16,rita_ptn:12
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AA = "ACDEFGHIKLMNPQRSTVWY"


def residue_aa_codes(layer_dir: Path):
    """Per-row amino-acid index (0..19), -1 for anything non-standard."""
    seqs = json.loads((layer_dir / "sequences.json").read_text())
    uids = json.loads((layer_dir / "uids.json").read_text())
    lengths = np.load(layer_dir / "lengths.npy")
    if isinstance(seqs, dict):
        ordered = [seqs[str(u)] for u in uids]
    else:
        ordered = list(seqs)
    idx = {a: i for i, a in enumerate(AA)}
    codes = np.full(int(np.sum(lengths)), -1, dtype=np.int8)
    off = 0
    for s, L in zip(ordered, lengths):
        L = int(L)
        for k, ch in enumerate(str(s)[:L]):
            codes[off + k] = idx.get(ch.upper(), -1)
        off += L
    return codes


def selectivity(Z, codes, chunk=20000):
    """(S_j, top1_share_j) per feature, background-corrected over the 20 AA types."""
    n_feat = int(Z.shape[1])
    mass = np.zeros((20, n_feat), dtype=np.float64)
    counts = np.zeros(20, dtype=np.float64)
    for i in range(0, Z.shape[0], chunk):
        z = np.asarray(Z[i:i + chunk], dtype=np.float32)
        c = codes[i:i + chunk]
        z = np.abs(z)                      # activation magnitude; TopK activations are >= 0
        for a in range(20):
            m = c == a
            if m.any():
                mass[a] += z[m].sum(axis=0)
                counts[a] += int(m.sum())
    # Background correction: mass per residue of that type, so abundance does not masquerade
    # as selectivity.
    rate = mass / np.maximum(counts[:, None], 1.0)
    tot = rate.sum(axis=0)
    ok = tot > 0
    p = np.zeros_like(rate)
    p[:, ok] = rate[:, ok] / tot[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=0)
    S = 1.0 - H / np.log(20)
    S[~ok] = np.nan
    top1 = np.where(ok, p.max(axis=0), np.nan)
    return S, top1, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_outlier")
    ap.add_argument("--cells", default="esm2_ptn:16,rita_ptn:12",
                    help="comma-separated model:layer")
    ap.add_argument("--out", default="results_rigor/aa_selectivity.csv")
    args = ap.parse_args()

    rows, per_feature = [], []
    for spec in [c.strip() for c in args.cells.split(",") if c.strip()]:
        model, layer = spec.split(":")
        d = Path(args.root) / model / f"layer_{layer}"
        zp, mp = d / "Z.npy", d / "struct_seq_metrics.csv"
        if not (zp.exists() and mp.exists()):
            print(f"  [skip] {spec}: missing {'Z.npy' if not zp.exists() else 'struct_seq_metrics.csv'}")
            continue
        Z = np.load(zp, mmap_mode="r")
        codes = residue_aa_codes(d)
        if len(codes) != Z.shape[0]:
            print(f"  [skip] {spec}: {len(codes)} residues vs {Z.shape[0]} Z rows")
            continue
        print(f"  {spec}: Z {Z.shape}, {int((codes >= 0).sum()):,} standard residues")
        S, top1, p = selectivity(Z, codes)
        met = pd.read_csv(mp)
        sd = met["struct_delta"].to_numpy(np.float64)
        qd = met["seq_delta"].to_numpy(np.float64) if "seq_delta" in met else None
        n = min(len(S), len(sd))
        S, top1, sd = S[:n], top1[:n], sd[:n]
        good = np.isfinite(S) & np.isfinite(sd)
        r_s, p_s = spearmanr(S[good], sd[good])
        r_t, p_t = spearmanr(top1[good], sd[good])
        rec = dict(cell=spec, n_features=int(good.sum()),
                   rho_selectivity_struct=r_s, p_struct=p_s,
                   rho_top1_struct=r_t, p_top1=p_t,
                   mean_selectivity=float(np.nanmean(S)))
        if qd is not None:
            qd = qd[:n]
            g2 = good & np.isfinite(qd)
            r_q, p_q = spearmanr(S[g2], qd[g2])
            rec.update(rho_selectivity_seq=r_q, p_seq=p_q)
        rows.append(rec)
        for j in range(n):
            per_feature.append(dict(cell=spec, feature=j, selectivity=S[j],
                                    top1_share=top1[j], struct_delta=sd[j],
                                    top_aa=AA[int(np.argmax(p[:, j]))] if np.isfinite(S[j]) else ""))
        del Z

    if not rows:
        raise SystemExit("no cells processed")
    r = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out, index=False)
    pd.DataFrame(per_feature).to_csv(args.out.replace(".csv", "_per_feature.csv"), index=False)

    print("\n" + "=" * 90)
    print("DOES THE CO-ACTIVATION METRIC TRACK AMINO-ACID SELECTIVITY?")
    print("=" * 90)
    print(r.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n  PREDICTION (paper §4.5): rho_selectivity_struct > 0 and material.")
    for _, x in r.iterrows():
        agree = np.sign(x.rho_selectivity_struct) == np.sign(x.rho_top1_struct)
        print(f"  {x.cell}: rho = {x.rho_selectivity_struct:+.3f} (p = {x.p_struct:.2g}); "
              f"crude measure agrees: {agree}")
        if "rho_selectivity_seq" in x and np.isfinite(x.rho_selectivity_seq):
            print(f"      control, sequential locality: rho = {x.rho_selectivity_seq:+.3f}")
    print(f"\n  wrote -> {args.out}")


if __name__ == "__main__":
    main()
