#!/usr/bin/env python3
"""
experiment_interplm_metric.py — Simon & Zou's clustering metric, implemented exactly.

WHY THIS EXISTS
---------------
The standing objection to this paper is: "the metric is introduced in this work,
which makes it unclear who the cautionary tale is for." Everything we have run so
far is OUR estimator, at most borrowing their contact definition. That cannot answer
the objection, because a failure of our estimator is a fact about our estimator.

This implements their procedure as specified in the Nature Methods Methods section,
verbatim, and runs it on the real and shuffled controlled models. If their metric
also scores shuffled-trained models higher, the cautionary tale has an audience and
the paper's central claim generalises beyond our variant. If it does not, we learn
that our anchor/threshold/aggregation is what breaks, which is equally worth knowing
and considerably narrows the paper.

THEIR PROCEDURE, quoted from the Methods
----------------------------------------
  "Use the SAE to get feature activation values ai for each amino acid position i
   Identify all positions where activations exceed threshold ai > 0.6
   Retain only proteins with complete AlphaFold 3D coordinate data and at least 25
   proteins where activations exceed threshold
   Randomly sample up to 100 proteins per feature for analysis"

  "for each selected protein: Find the position with maximum activation
   posmax = arg max ai
   Sequential clustering: seq_score = mean activation over
     seq_neighbors = {j : |j - posmax| <= 2, j != posmax}
   Structural clustering: struct_score = mean activation over
     struct_neighbors = {j : distance(j, posmax) <= 6 A, j != posmax}
   Distance calculated using Ca coordinates"

  "Create 5 random permutations of the activation values ... recalculate
   seq_score_null and struct_score_null using the same neighbor positions but
   shuffled activation values ... Average across the five permutations"

  "Assessed clustering significance using paired t-tests and Cohen's d effect sizes
   across 100 proteins per feature"

WHAT DIFFERS FROM OUR L_struct, and is therefore what this run controls for
  anchor        one highest-activation residue per protein   (we: every active residue)
  gate          absolute activation > 0.6                    (we: top-10% fraction)
  d is over     proteins within a feature                    (we: features)
  selection     Bonferroni-corrected structural p < 0.05     (we: none)
  reported as   structural-to-sequential ratio               (we: structural alone)
  separation    none                                         (we: |i-j| >= 12)

NOTE ON THE 0.6 THRESHOLD. The Methods give it as an absolute value. Our own S2.1
records it as applying after each feature is normalised to [0,1], which is the only
reading under which a fixed 0.6 is scale-free across features. Both are available
here; --gate-mode raw is the literal text, --gate-mode normalised is the default.
Report which was used -- it changes how many features qualify.

USAGE
  python experiment_interplm_metric.py --layer-dir outputs_ctrl/ckpt_clm_s42/layer_14 \
      --out results_interplm_metric/clm_s42_L14.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from cpu_stage import load_ref_seqs, extract_chain_ca_seq_coords

from Bio.PDB import PDBParser
from Bio import pairwise2


def ca_coords_for(uid, Lr, ref_seq, pdb_dir):
    """Ca coordinates in REFERENCE-sequence indexing, NaN where unresolved.

    Mirrors the alignment in cpu_stage._process_single_protein_graph so the residue
    indexing is identical to every other number in this project.
    """
    Lr = int(Lr)
    out = np.full((Lr, 3), np.nan, dtype=np.float32)
    p = Path(pdb_dir) / f"{str(uid)[1:5].lower()}.pdb"
    if not p.exists():
        return out
    try:
        struct = PDBParser(QUIET=True).get_structure("t", str(p))
        best_score, best = None, None
        for model in struct:
            for chain in model:
                cseq, ccoord = extract_chain_ca_seq_coords(chain)
                if len(cseq) < 10:
                    continue
                aln = pairwise2.align.globalms(ref_seq, cseq, 2, -1, -5, -0.5,
                                               one_alignment_only=True)
                if not aln:
                    continue
                aln = aln[0]
                cur = np.full((Lr, 3), np.nan, dtype=np.float32)
                ri = ci = -1
                for a, b in zip(aln.seqA, aln.seqB):
                    if a != "-":
                        ri += 1
                    if b != "-":
                        ci += 1
                    if a != "-" and b != "-" and 0 <= ri < Lr and 0 <= ci < len(ccoord):
                        cur[ri] = ccoord[ci]
                if best_score is None or aln.score > best_score:
                    best_score, best = aln.score, cur
            break
    except Exception:
        return out
    return best if best is not None else out


def feature_block(Z, f0, f1, coords, spans, rng_seed, gate, min_prot, max_prot,
                  cutoff, n_perm, normalise=True):
    """Their metric for features [f0, f1)."""
    rng = np.random.default_rng(rng_seed + f0)
    rows = []
    for f in range(f0, f1):
        a_all = np.asarray(Z[:, f], dtype=np.float32)
        # Their "Feature normalization": each feature is divided by its maximum over
        # the WHOLE corpus, so every feature lies in [0,1], and the 0.6 gate is then
        # absolute on that scale. Dividing by the per-protein max instead (which the
        # first version did) is a third thing matching neither reading -- for a
        # feature whose corpus max is 28.4 it makes the effective threshold 17.0.
        if normalise:
            fmax = float(a_all.max())
            if fmax > 0:
                a_all = a_all / fmax
        seq_o, str_o, seq_n, str_n = [], [], [], []
        qualifying = []
        for k, (lo, hi) in enumerate(spans):
            a = a_all[lo:hi]
            if coords[k] is None or not np.isfinite(a).all() or a.max() <= 0:
                continue
            if (a > gate).sum() < 1:      # a is already globally normalised
                continue
            qualifying.append(k)
        if len(qualifying) < min_prot:                 # their >= 25 proteins rule
            rows.append(dict(feature=f, n_proteins=len(qualifying), d_struct=np.nan,
                             p_struct=np.nan, d_seq=np.nan, ratio=np.nan))
            continue
        if len(qualifying) > max_prot:                 # their <= 100 sample rule
            qualifying = list(rng.choice(qualifying, max_prot, replace=False))

        for k in qualifying:
            lo, hi = spans[k]
            a = a_all[lo:hi]
            C = coords[k]
            pm = int(np.argmax(a))
            if not np.isfinite(C[pm]).all():
                continue
            L = hi - lo
            sn = [j for j in range(max(0, pm - 2), min(L, pm + 3)) if j != pm]
            d = np.linalg.norm(C - C[pm], axis=1)
            st = [j for j in range(L) if j != pm and np.isfinite(d[j]) and d[j] <= cutoff]
            if not sn or not st:
                continue
            seq_o.append(float(a[sn].mean())); str_o.append(float(a[st].mean()))
            sn_null, st_null = [], []
            for _ in range(n_perm):                    # permute VALUES, keep positions
                ap = rng.permutation(a)
                sn_null.append(float(ap[sn].mean())); st_null.append(float(ap[st].mean()))
            seq_n.append(float(np.mean(sn_null))); str_n.append(float(np.mean(st_null)))

        if len(str_o) < 3:
            rows.append(dict(feature=f, n_proteins=len(str_o), d_struct=np.nan,
                             p_struct=np.nan, d_seq=np.nan, ratio=np.nan))
            continue
        so, sn_ = np.array(str_o), np.array(str_n)
        qo, qn = np.array(seq_o), np.array(seq_n)
        # Cohen's d across PROTEINS within this feature -- their standardisation
        ds = (so - sn_).mean() / ((so - sn_).std(ddof=1) + 1e-12)
        dq = (qo - qn).mean() / ((qo - qn).std(ddof=1) + 1e-12)
        p = float(stats.ttest_rel(so, sn_).pvalue)
        rows.append(dict(feature=f, n_proteins=len(so), d_struct=float(ds),
                         p_struct=p, d_seq=float(dq),
                         ratio=float(ds / dq) if abs(dq) > 1e-9 else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--cutoff", type=float, default=6.0)      # their 6 A
    ap.add_argument("--gate", type=float, default=0.6)        # their > 0.6
    ap.add_argument("--gate-mode", choices=["global", "raw"], default="global",
                    help="global: divide each feature by its corpus-wide max, then "
                         "apply --gate absolutely -- this is what the Methods describe. "
                         "raw: apply --gate to unnormalised activations.")
    ap.add_argument("--min-proteins", type=int, default=25)   # their >= 25
    ap.add_argument("--max-proteins", type=int, default=100)  # their <= 100
    ap.add_argument("--n-perm", type=int, default=5)          # their 5
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.layer_dir)
    Z = np.asarray(np.load(d / "Z.npy"), dtype=np.float32)
    uids = [str(u) for u in json.loads((d / "uids.json").read_text())]
    lengths = np.load(d / "lengths.npy")
    ref_seqs = load_ref_seqs(d)
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    spans = [(int(o), int(o) + int(L)) for o, L in zip(offs, lengths)]

    print(f"  cell   : {d}")
    print(f"  spec   : {args.cutoff} A, gate {args.gate} ({args.gate_mode}), "
          f">={args.min_proteins} proteins, <={args.max_proteins} sampled, "
          f"{args.n_perm} permutations, NO separation floor")
    print(f"  parsing Ca coordinates for {len(uids)} proteins ...")
    coords = Parallel(n_jobs=args.n_jobs)(
        delayed(ca_coords_for)(u, L, ref_seqs[u], args.pdb_dir)
        for u, L in zip(uids, lengths))
    coords = [c if np.isfinite(c).any() else None for c in coords]
    print(f"  {sum(c is not None for c in coords)} with usable coordinates")

    gate = args.gate
    nF = Z.shape[1] if args.max_features == 0 else min(args.max_features, Z.shape[1])
    step = 64
    out = Parallel(n_jobs=args.n_jobs)(
        delayed(feature_block)(Z, f, min(f + step, nF), coords, spans, args.seed,
                               gate, args.min_proteins, args.max_proteins,
                               args.cutoff, args.n_perm,
                               args.gate_mode == "global")
        for f in range(0, nF, step))
    df = pd.DataFrame([r for blk in out for r in blk])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df.dropna(subset=["d_struct"])
    if len(ok):
        bonf = 0.05 / max(len(ok), 1)
        sig = ok[ok.p_struct < bonf]                  # their selection criterion
        print(f"\n  features scored           : {len(ok)} / {nF}")
        print(f"  significant structural    : {len(sig)}  "
              f"(Bonferroni p < {bonf:.2e})")
        print(f"  mean d_struct over scored : {ok.d_struct.mean():+.4f}")
        print(f"  mean d_struct over sig    : "
              f"{sig.d_struct.mean():+.4f}" if len(sig) else "  (none significant)")
        if len(sig):
            print(f"  median struct/seq ratio   : {sig.ratio.median():+.4f}")
    print(f"\n  written: {args.out}")
    print("  THE COMPARISON: run this on the real and the shuffled tree. If the")
    print("  shuffled models score higher on mean d_struct, or produce more")
    print("  significant structural features, the failure is not specific to our")
    print("  variant and the cautionary tale has an audience.")


if __name__ == "__main__":
    main()
