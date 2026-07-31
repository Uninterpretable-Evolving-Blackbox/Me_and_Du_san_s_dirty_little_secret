#!/usr/bin/env python3
"""
experiment_contact_def_sweep.py — does the result depend on how a "contact" is defined?

THE OBJECTION (the project's own R12, currently owned but never tested on this pair)
  L_struct defines a contact as Ca-Ca < 8 A with |i-j| >= 12. The field convention for
  long-range contact evaluation is Cb-Cb < 8 A with |i-j| >= 24. The project owns the
  deviation honestly and swept it for the ESM-2/RITA pair — but never for the CONTROLLED
  pair, which is the comparison carrying the causal claim. So the one place the claim is
  identified is the one place the contact definition is untested.

  This closes that, and works on any pair of layer dirs, so it also re-runs the family
  pair if you point it there.

THE GRID (matched to the family-pair sweep, plus the convention the project deviates from)
  cutoff  in {6, 8, 10} A     — the family sweep's grid; 8 is the project default
  seq_gap in {6, 12, 24}      — 12 is the project default, 24 is the field convention,
                                6 is the loose end used by some contact benchmarks
  9 cells per layer dir. The (8, 12) cell must reproduce the existing
  struct_seq_metrics.csv exactly; that agreement is printed as a self-check.

WHAT WOULD FAIL THE TEST
  Not a change in magnitude — a looser gap admits easier contacts and shifts every
  number. The claim fails if the SIGN of the cross-model d flips, or if it survives only
  at (8, 12). In particular, if the effect disappears at |i-j| >= 24, the paper cannot
  describe it as a long-range structural effect and must say so.

NOTE: Cb is NOT swept. Every contact in this codebase is Ca-based, all the way down to
the neighbour-graph builder; switching to Cb is a change to the structure parser, not a
parameter, and doing it half-way would silently mix definitions. Stated as a limitation
rather than faked.

Usage
  .venv/bin/python experiment_contact_def_sweep.py \
      --root outputs_ctrl \
      --model-a ckpt_mlm_s42_token --model-b ckpt_clm_s42 \
      --layers 11,14,18
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cpu_stage import (
    load_layer, load_ref_seqs, build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations, _process_struct_seq_chunk_v3,
)

CUTOFFS = (6.0, 8.0, 10.0)      # default grid; override with --cutoffs
SEQ_GAPS = (6, 12, 24)          # default grid; override with --gaps
DEFAULT_CELL = (8.0, 12)        # the project default cell, used for the self-check

# NOTE (2026-07-30): --cutoffs/--gaps were added so the metric can be evaluated at
# OTHER published settings, in particular Simon & Zou (InterPLM): 6 A with NO
# minimum sequence separation. Pass --gaps 1 for "no floor" (|i-j| >= 1 excludes
# only the residue itself). Defaults are unchanged, so existing invocations and
# every number already delivered reproduce bit-for-bit.


def struct_delta_for(layer_dir: Path, pdb_dir: Path, cutoff, seq_gap,
                     n_shuffles=5, n_jobs=-1, topk_frac=0.10):
    """Per-feature struct_delta at one (cutoff, seq_gap). Needs Z.npy."""
    Z, uids, lengths = load_layer(layer_dir)
    uids = [str(u) for u in uids]
    ref_seqs = load_ref_seqs(layer_dir)
    n_res = int(np.sum(lengths))

    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, lengths, ref_seqs, pdb_dir, n_jobs,
        contact_cutoff=cutoff, seq_gap_min=seq_gap)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)
    del seq_adj, struct_adj

    perms = build_protein_permutations(lengths, n_shuffles)
    n_features = int(Z.shape[1])
    chunk = 256
    n_chunks = (n_features + chunk - 1) // chunk
    res = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk, Z, None, A_seq, deg_seq, A_struct, deg_struct,
            perms, n_features, topk_frac)
        for ci in range(n_chunks))
    idx = np.concatenate([r[0] for r in res])
    obs = np.concatenate([r[2] for r in res])
    sh = np.concatenate([r[4] for r in res])
    order = np.argsort(idx)
    return (obs - sh)[order], int(A_struct.nnz)


def cohens_d(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_ctrl")
    ap.add_argument("--model-a", required=True, help="the arm expected to score HIGHER")
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--layers", default="11,14,18")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default="results_contact_def_sweep")
    ap.add_argument("--cutoffs", default=None,
                    help="comma-separated Ca-Ca cutoffs in Angstrom. "
                         "Default: 6,8,10. InterPLM setting: 6")
    ap.add_argument("--gaps", default=None,
                    help="comma-separated minimum |i-j|. Default: 6,12,24. "
                         "InterPLM setting: 1 (no separation floor)")
    args = ap.parse_args()

    cutoffs = tuple(float(x) for x in args.cutoffs.split(",")) if args.cutoffs else CUTOFFS
    gaps    = tuple(int(x)   for x in args.gaps.split(","))    if args.gaps    else SEQ_GAPS
    if any(g < 1 for g in gaps):
        raise SystemExit("--gaps must be >= 1; |i-j| >= 1 already excludes the residue itself")
    print(f"  grid: cutoffs={cutoffs} A  x  gaps={gaps}  "
          f"({len(cutoffs)*len(gaps)} cells per layer)")

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)
    rows = []

    for layer in layers:
        da = Path(args.root) / args.model_a / f"layer_{layer}"
        db = Path(args.root) / args.model_b / f"layer_{layer}"
        if not ((da / "Z.npy").exists() and (db / "Z.npy").exists()):
            print(f"  [skip] layer {layer}: Z.npy missing "
                  f"({'A ' if not (da/'Z.npy').exists() else ''}"
                  f"{'B' if not (db/'Z.npy').exists() else ''}). "
                  f"Re-run the extractor with KEEP_Z=1.")
            continue
        for cutoff in cutoffs:
            for gap in gaps:
                sa, nnz_a = struct_delta_for(da, pdb_dir, cutoff, gap,
                                             args.n_shuffles, args.n_jobs)
                sb, _ = struct_delta_for(db, pdb_dir, cutoff, gap,
                                         args.n_shuffles, args.n_jobs)
                d = cohens_d(sa, sb)
                is_default = (cutoff, gap) == DEFAULT_CELL
                rec = dict(layer=layer, cutoff_A=cutoff, seq_gap=gap, d=d,
                           mean_a=float(sa.mean()), mean_b=float(sb.mean()),
                           n_contact_edges=nnz_a, is_project_default=is_default)
                if is_default:
                    ref = da / "struct_seq_metrics.csv"
                    if ref.exists():
                        r = pd.read_csv(ref)["struct_delta"].to_numpy(np.float64)
                        if len(r) == len(sa):
                            rec["selfcheck_max_abs_diff"] = float(np.max(np.abs(r - sa)))
                rows.append(rec)
                tag = "  <- project default" if is_default else ""
                print(f"  L{layer} Ca<{cutoff:<4} |i-j|>={gap:<3} d={d:+.4f} "
                      f"edges={nnz_a:,}{tag}")

    if not rows:
        raise SystemExit("nothing computed — check --root/--layers and that Z.npy exists")
    df = pd.DataFrame(rows)
    df.to_csv(out / "contact_def_sweep.csv", index=False)

    print("\n" + "=" * 88)
    print(f"CONTACT-DEFINITION SWEEP — d({args.model_a} - {args.model_b})")
    print("=" * 88)
    piv = df.pivot_table(index="seq_gap", columns="cutoff_A", values="d", aggfunc="mean")
    print(piv.to_string(float_format=lambda v: f"{v:+.4f}"))
    v = df["d"].to_numpy(float); v = v[np.isfinite(v)]
    print(f"\n  positive at {int((v > 0).sum())}/{v.size} cells; "
          f"range [{v.min():+.4f}, {v.max():+.4f}]")
    g24 = df[df.seq_gap == 24]["d"].to_numpy(float); g24 = g24[np.isfinite(g24)]
    if g24.size:
        print(f"  at the FIELD CONVENTION |i-j| >= 24: mean d {g24.mean():+.4f}, "
              f"positive at {int((g24 > 0).sum())}/{g24.size}")
        if (g24 > 0).mean() < 0.5:
            print("  !! The effect does NOT hold at the conventional separation. The paper cannot")
            print("  !! call this a long-range structural effect without saying this explicitly.")
    sc = df.get("selfcheck_max_abs_diff")
    if sc is not None and sc.notna().any():
        print(f"  self-check at (8, 12) vs struct_seq_metrics.csv: "
              f"max|diff| = {sc.dropna().max():.2e}")
    print(f"\n  wrote -> {out}/contact_def_sweep.csv")


if __name__ == "__main__":
    main()
