#!/usr/bin/env python3
"""
rescore_denominator.py — L_struct and L_seq under alternative denominators.

WHY
---
L_struct divides by SD(a_f), the standard deviation of feature f's raw
activations. SD is scale-dependent, and order-destroyed dictionaries have
activation scales roughly an order of magnitude below their native
counterparts, so the rise reported under the corpus control could in principle
be a denominator artefact. The paper lists this as untested.

It cannot be tested from the outputs already on disk. `struct_seq_metrics.csv`
stores only the ratio: every column in it has already been divided by
SD(a_f) + 1e-6 inside cpu_stage._cohens_d_vectorized, and the per-feature SD is
never written out. Recovering the numerator therefore requires a second pass
over Z.npy. That is what this script does.

WHAT IT DOES NOT DO
-------------------
It does not modify cpu_stage.py, and it never writes struct_seq_metrics.csv.
Every published number came out of cpu_stage.py unchanged, and it stays that
way. This script IMPORTS the graph construction, the permutation draw and the
adjacency conversion from cpu_stage, so the contact graph, the null and the
active-set rule are the same objects the paper used, not reimplementations.

THE SELF-CHECK IS THE POINT
---------------------------
The script always recomputes the published `sd` denominator alongside the new
ones and compares it against the existing struct_seq_metrics.csv, feature by
feature. If the reproduction is not exact the script EXITS NON-ZERO and writes
nothing. A denominator variant is only worth reading if the pipeline that
produced it reproduces the number it is being compared against.

DENOMINATORS
------------
  sd     SD(a_f) + 1e-6                published; used for the self-check
  fixed  1.0                           numerator in neighbour-averaged units
  iqr    IQR(a_f over nonzeros) + 1e-6 robust scale; falls back to sd where
                                       a feature has no nonzero spread (flagged)
  rank   SD of the rank transform      activations replaced, per feature, by
                                       within-feature ranks scaled to [0, 1];
                                       a genuinely rank-based variant, so the
                                       numerator changes too. Needs a second
                                       matmul pass, hence its own --denominators
                                       entry rather than being free.

USAGE
    python rescore_denominator.py --layer-dir outputs_ctrl/ckpt_clm_s42/layer_11
    python rescore_denominator.py --layer-dir <dir> --denominators sd,fixed,rank
    python rescore_denominator.py --layer-dir <dir> --no-self-check   # NOT ADVISED

OUTPUT
    <layer-dir>/struct_seq_metrics_denominators.csv
        feature_idx,
        denom_sd, denom_iqr, denom_rank_sd,        (the divisors themselves)
        n_active, iqr_degenerate,
        {seq,struct}_delta_{mode} for each mode requested
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Imported, never modified. If this import fails the environment is wrong and
# nothing below would be comparable to the published numbers anyway.
import cpu_stage as cs
from joblib import Parallel, delayed, cpu_count

EPS = 1e-6
MIN_ACTIVE = 5          # cpu_stage zeroes d for features with fewer than this
ALL_MODES = ("sd", "fixed", "iqr", "rank")


# ---------------------------------------------------------------------------
#   numerator: cpu_stage._cohens_d_vectorized with the division factored out
# ---------------------------------------------------------------------------
def _numerator(acts_chunk, A_sp, deg, topk_frac):
    """
    (active_mean - global_mean) of the neighbour-averaged activations.

    Mirrors cpu_stage._cohens_d_vectorized exactly up to the final division,
    including the strict `>` on the percentile threshold and the
    n_active < 5 -> 0 rule (applied to the numerator, which is equivalent
    since 0 / x == 0 for every finite denominator used here).
    """
    n_res, n_feat = acts_chunk.shape

    nbr_sums = np.asarray(A_sp @ acts_chunk, dtype=np.float32)
    has_nbrs = deg > 0
    nbr_sums[has_nbrs] /= deg[has_nbrs, None]
    nbr_sums[~has_nbrs] = 0.0
    smoothed = nbr_sums

    global_mean = smoothed.mean(axis=0)

    thresh = np.percentile(acts_chunk, 100.0 * (1.0 - topk_frac), axis=0)
    active = acts_chunk > thresh[None, :]

    n_active = active.sum(axis=0).astype(np.float32)
    active_sum = (smoothed * active).sum(axis=0)
    n_safe = n_active.copy()
    n_safe[n_safe == 0] = 1.0
    active_mean = active_sum / n_safe

    num = (active_mean - global_mean).astype(np.float32)
    num[n_active < MIN_ACTIVE] = 0.0
    return num, n_active


def _denominators(acts, modes):
    """Per-feature divisors. Returns (dict mode -> vector, extras dict)."""
    out, extras = {}, {}
    gstd = np.std(acts, axis=0).astype(np.float32)
    extras["denom_sd"] = gstd
    if "sd" in modes:
        out["sd"] = gstd + EPS
    if "fixed" in modes:
        out["fixed"] = np.ones_like(gstd)
    if "iqr" in modes:
        # IQR over the nonzero activations only: with a TopK dictionary most
        # features are exactly 0 at both quartiles, and an all-zero IQR would
        # turn +1e-6 into a divide-by-nothing.
        iqr = np.zeros_like(gstd)
        degenerate = np.zeros(acts.shape[1], dtype=bool)
        for j in range(acts.shape[1]):
            col = acts[:, j]
            nz = col[col > 0]
            if nz.size < 4:
                degenerate[j] = True
                continue
            q75, q25 = np.percentile(nz, [75.0, 25.0])
            iqr[j] = np.float32(q75 - q25)
        degenerate |= (iqr <= 0)
        # fall back to SD where IQR is undefined, and say so in the output
        iqr = np.where(degenerate, gstd, iqr).astype(np.float32)
        out["iqr"] = iqr + EPS
        extras["denom_iqr"] = iqr
        extras["iqr_degenerate"] = degenerate
    return out, extras


def _rank_transform(acts):
    """
    Within-feature ranks scaled to [0, 1], average ranks for ties.

    Ties matter here rather than being a detail: a TopK dictionary leaves ~90%
    of each column at exactly 0, so the zeros form one enormous tied block and
    every one of them must get the same rank. scipy.stats.rankdata does this in
    C along an axis; the equivalent Python loop is ~750M iterations at the real
    feature-matrix size and is not an option.
    """
    from scipy.stats import rankdata
    n = acts.shape[0]
    r = rankdata(acts, method="average", axis=0).astype(np.float32)  # 1..n
    return ((r - 1.0) / max(n - 1, 1)).astype(np.float32)


def _process_chunk(ci, chunk_size, Z, A_proj, A_seq, deg_seq, A_struct,
                   deg_struct, perm_indices, n_features, topk_frac, modes):
    i = ci * chunk_size
    end = min(i + chunk_size, n_features)

    if A_proj is None:
        acts = np.asarray(Z[:, i:end], dtype=np.float32)
    else:
        acts = np.asarray(A_proj @ np.asarray(Z[:, i:end], dtype=np.float32),
                          dtype=np.float32)

    plain = tuple(m for m in modes if m != "rank")
    denoms, extras = _denominators(acts, plain)

    res = {"idx": np.arange(i, end, dtype=np.int32)}
    for k, v in extras.items():
        res[k] = v

    def run(a, denom_map, suffix_modes):
        seq_o, n_act = _numerator(a, A_seq, deg_seq, topk_frac)
        str_o, _ = _numerator(a, A_struct, deg_struct, topk_frac)
        acc = {m: [np.zeros(end - i, np.float32), np.zeros(end - i, np.float32)]
               for m in suffix_modes}
        for perm in perm_indices:
            ap = a[perm]
            sq, _ = _numerator(ap, A_seq, deg_seq, topk_frac)
            st, _ = _numerator(ap, A_struct, deg_struct, topk_frac)
            for m in suffix_modes:
                # accumulate the NORMALISED values, in cpu_stage's order, so
                # the sd mode reproduces its floating-point result exactly
                acc[m][0] += sq / denom_map[m]
                acc[m][1] += st / denom_map[m]
        n_sh = len(perm_indices)
        for m in suffix_modes:
            if n_sh > 0:
                acc[m][0] /= n_sh
                acc[m][1] /= n_sh
            res[f"seq_delta_{m}"] = (seq_o / denom_map[m]) - acc[m][0]
            res[f"struct_delta_{m}"] = (str_o / denom_map[m]) - acc[m][1]
        return n_act

    n_act = run(acts, denoms, plain)
    res["n_active"] = n_act

    if "rank" in modes:
        ranked = _rank_transform(acts)
        rstd = np.std(ranked, axis=0).astype(np.float32)
        res["denom_rank_sd"] = rstd
        run(ranked, {"rank": rstd + EPS}, ("rank",))

    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--denominators", default="sd,fixed,iqr,rank",
                    help=f"comma-separated subset of {','.join(ALL_MODES)}; "
                         "'sd' is always added because the self-check needs it")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--n-shuffles", type=int, default=5,
                    help="MUST match the run being reproduced (the paper uses 5)")
    ap.add_argument("--contact-cutoff", type=float, default=cs.DEFAULT_CONTACT_CUTOFF)
    ap.add_argument("--seq-gap-min", type=int, default=cs.DEFAULT_SEQ_GAP_MIN)
    ap.add_argument("--topk-frac", type=float, default=cs.DEFAULT_TOPK_FRAC)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--out-name", default="struct_seq_metrics_denominators.csv")
    ap.add_argument("--self-check-tol", type=float, default=1e-5)
    ap.add_argument("--no-self-check", action="store_true",
                    help="skip reproduction of the published column. Do not use "
                         "for anything that goes in the paper.")
    args = ap.parse_args()

    modes = [m.strip() for m in args.denominators.split(",") if m.strip()]
    bad = [m for m in modes if m not in ALL_MODES]
    if bad:
        raise SystemExit(f"unknown denominator(s): {bad}; choose from {ALL_MODES}")
    if "sd" not in modes:
        modes.insert(0, "sd")
    modes = [m for m in ALL_MODES if m in modes]     # canonical order

    layer_dir = Path(args.layer_dir)
    if not (layer_dir / "Z.npy").exists():
        raise SystemExit(f"no Z.npy in {layer_dir} — run STAGE=1 bash RUN_MUSTRUNS.sh first")

    ref_csv = layer_dir / "struct_seq_metrics.csv"
    if not args.no_self_check and not ref_csv.exists():
        raise SystemExit(
            f"no {ref_csv} to self-check against.\n"
            "  Either run cpu_stage.py on this cell first, or pass --no-self-check\n"
            "  and accept that the output is unverified.")

    print("=" * 68)
    print("rescore_denominator.py")
    print(f"  layer dir:    {layer_dir}")
    print(f"  denominators: {', '.join(modes)}")
    print(f"  n_shuffles:   {args.n_shuffles}   (must match the reference run)")
    print(f"  contact:      {args.contact_cutoff} A, gap >= {args.seq_gap_min}, "
          f"topk {args.topk_frac}")
    print("=" * 68)

    Z, uids, tok_lengths = cs.load_layer(layer_dir)
    ref_seqs = cs.load_ref_seqs(layer_dir)
    res_lengths = np.array([len(ref_seqs[u]) for u in uids], dtype=np.int32)
    if int(np.sum(res_lengths)) != int(Z.shape[0]):
        raise SystemExit("sum(len(seq)) != Z rows — this is not a residue-model layer dir")
    n_res, n_features = int(Z.shape[0]), int(Z.shape[1])
    print(f"  Z: {Z.shape}")

    n_jobs = args.n_jobs if args.n_jobs > 0 else cpu_count()
    seq_adj, struct_adj = cs.build_neighbor_graphs_residue_parallel(
        uids, res_lengths, ref_seqs, Path(args.pdb_dir), n_jobs,
        contact_cutoff=args.contact_cutoff, seq_gap_min=args.seq_gap_min)
    A_seq, deg_seq = cs.adj_list_to_sparse(seq_adj, n_res)
    A_struct, deg_struct = cs.adj_list_to_sparse(struct_adj, n_res)
    del seq_adj, struct_adj
    print(f"  edges: seq {A_seq.nnz:,}  struct {A_struct.nnz:,}")

    # identical draw to cpu_stage (same function, same default seed)
    perm_indices = cs.build_protein_permutations(res_lengths, args.n_shuffles)

    chunk_size = args.chunk_size
    mem_per_worker = n_res * chunk_size * 4 * 5 / 1e9
    mem_budget_gb = float(os.environ.get("CPU_STAGE_MEM_GB", 100.0))
    max_safe = max(1, int(mem_budget_gb / max(mem_per_worker, 0.1)))
    eff_jobs = min(n_jobs, max_safe)
    n_chunks = (n_features + chunk_size - 1) // chunk_size
    print(f"  {n_features} features in {n_chunks} chunks, {eff_jobs} workers")

    results = Parallel(n_jobs=eff_jobs, verbose=5)(
        delayed(_process_chunk)(
            ci, chunk_size, Z, None, A_seq, deg_seq, A_struct, deg_struct,
            perm_indices, n_features, args.topk_frac, tuple(modes))
        for ci in range(n_chunks))

    idx = np.concatenate([r["idx"] for r in results])
    order = np.argsort(idx)
    cols = {"feature_idx": idx[order].astype(np.int32)}
    for key in results[0]:
        if key == "idx":
            continue
        cols[key] = np.concatenate([r[key] for r in results])[order]
    df = pd.DataFrame(cols)

    # ---------------------------------------------------------------- self-check
    status = "SKIPPED"
    if not args.no_self_check:
        ref = pd.read_csv(ref_csv)
        if len(ref) != len(df):
            raise SystemExit(
                f"SELF-CHECK FAILED: reference has {len(ref)} features, "
                f"this run has {len(df)}. Different dictionary — refusing to write.")
        worst = 0.0
        for col in ("seq_delta", "struct_delta"):
            d = float(np.max(np.abs(ref[col].to_numpy() - df[f"{col}_sd"].to_numpy())))
            worst = max(worst, d)
            print(f"  self-check max|{col} - {col}_sd| = {d:.3e}")
        if worst > args.self_check_tol:
            raise SystemExit(
                f"SELF-CHECK FAILED: max diff {worst:.3e} > tol {args.self_check_tol:.0e}.\n"
                "  The sd column does not reproduce struct_seq_metrics.csv, so the\n"
                "  other denominators are not comparable to the published numbers.\n"
                "  Most likely cause: --n-shuffles differs from the reference run\n"
                "  (cpu_stage.py DEFAULTS TO 3; the paper uses 5), or a different\n"
                "  --contact-cutoff / --seq-gap-min / --topk-frac. Nothing written.")
        status = f"PASSED (max diff {worst:.3e})"
        print(f"  self-check {status}")

    out = layer_dir / args.out_name
    df.to_csv(out, index=False)

    print("-" * 68)
    print(f"  mean struct_delta by denominator ({len(df)} features):")
    for m in modes:
        print(f"    {m:<6s} {df[f'struct_delta_{m}'].mean():+.6f}")
    if "iqr_degenerate" in df.columns:
        nd = int(df["iqr_degenerate"].sum())
        print(f"  iqr fell back to sd for {nd}/{len(df)} features")
    print(f"  self-check: {status}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
