#!/usr/bin/env python3
"""
analyze_rigor.py — the analysis layer for the rigor program (PREPRINT_RIGOR_PROGRAM.md).

Four things, none of which require re-running any model:

  A6/C7  frozen-vs-trained comparison. Does a RANDOM dictionary reproduce the ESM-2 > RITA
         effect? (Korznikov et al. 2026 showed frozen-component SAEs match trained ones on
         interpretability / probing / causal editing, so this is the control the project has
         never run.)
  A7     architecture robustness: TopK vs BatchTopK.
  E5     partial Spearman — the rank-gap guard for the crosscoder (R5).
  R9     multiplicity: Benjamini-Hochberg FDR across the 9 depths, plus a cluster-bootstrap
         OMNIBUS test of the pooled across-depth effect. The project currently reports 9
         independent per-depth CIs with no global control, which a statistically-minded
         reviewer will flag immediately.

Run:  .venv/bin/python analyze_rigor.py            # everything available on disk
      .venv/bin/python analyze_rigor.py --only a6
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("0%", 0, 0), ("13%", 4, 3), ("25%", 8, 6), ("38%", 12, 9), ("50%", 16, 12),
         ("63%", 20, 15), ("75%", 24, 18), ("88%", 28, 21), ("100%", 32, 23)]


# ----------------------------------------------------------------------------------
#                                  shared helpers
# ----------------------------------------------------------------------------------

def cohens_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def _rank(x):
    """Average-rank transform with NaN passthrough."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    out[ok] = pd.Series(x[ok]).rank(method="average").values
    return out


def partial_spearman(x, y, Z):
    """Spearman correlation of x and y after linearly partialling out the columns of Z.

    Rank-transform everything, then residualise ranked x and ranked y on ranked Z by least
    squares and take the Pearson correlation of the residuals. This is the standard
    rank-based partial correlation, and it is what separates a genuine Delta_norm <-> L_struct
    association from one induced by a capacity imbalance between the two arms.
    """
    x, y = _rank(x), _rank(y)
    Z = np.column_stack([_rank(Z[:, j]) for j in range(np.atleast_2d(Z).shape[1])]) \
        if np.asarray(Z).ndim > 1 else _rank(Z).reshape(-1, 1)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if ok.sum() < 10:
        return np.nan
    x, y, Z = x[ok], y[ok], Z[ok]
    A = np.column_stack([np.ones(len(x)), Z])
    # lstsq residuals; rcond=None uses the stable default cutoff
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    sx, sy = rx.std(), ry.std()
    return float(np.mean(rx * ry) / (sx * sy)) if sx > 0 and sy > 0 else np.nan


def benjamini_hochberg(p, alpha=0.05):
    """BH step-up. Returns (rejected mask, q-values)."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]        # enforce monotonicity
    out_q = np.empty(n); out_q[order] = np.clip(q, 0, 1)
    return out_q <= alpha, out_q


# ----------------------------------------------------------------------------------
#              A6 / A7 — frozen-vs-trained and architecture robustness
# ----------------------------------------------------------------------------------

def _sd(root, model, tag, L):
    p = Path(root) / f"{model}_{tag}" / f"layer_{L}" / "struct_seq_metrics.csv"
    return pd.read_csv(p)["struct_delta"].values if p.exists() else None


def _meta(root, model, tag, L):
    p = Path(root) / f"{model}_{tag}" / f"layer_{L}" / "META.json"
    return json.loads(p.read_text()) if p.exists() else None


def compare_variants(root="outputs_outlier", base="ptn",
                     variants=("ptn_batchtopk", "ptn_frozen_decoder",
                               "ptn_soft_frozen_decoder", "ptn_frozen_encoder")):
    """H1 Cohen's d per depth, for the trained arm and every variant that exists on disk."""
    rows = []
    for lab, le, lr in PAIRS:
        rec = {"depth": lab}
        b_e, b_r = _sd(root, "esm2", base, le), _sd(root, "rita", base, lr)
        rec["d_trained"] = cohens_d(b_e, b_r) if b_e is not None and b_r is not None else np.nan
        for v in variants:
            e, r = _sd(root, "esm2", v, le), _sd(root, "rita", v, lr)
            rec[f"d_{v.replace(base + '_', '')}"] = (cohens_d(e, r)
                                                     if e is not None and r is not None else np.nan)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.drop(columns=["depth"]).notna().sum().sum() == 0:
        return None
    return df


def report_variants(df):
    print("=" * 88)
    print("A6/A7  H1 Cohen's d (ESM-2 - RITA) by depth: TRAINED vs NULL BASELINES / ARCHITECTURES")
    print("=" * 88)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    cols = [c for c in df.columns if c.startswith("d_") and c != "d_trained"]
    print("-" * 88)
    base = df["d_trained"]
    print(f"  {'arm':<26}{'mean d':>9}{'9/9 pos':>9}{'retained vs trained':>22}")
    print(f"  {'trained (TopK)':<26}{base.mean():>+9.3f}{int((base > 0).sum()):>6}/9{'—':>22}")
    for c in cols:
        v = df[c]
        if v.isna().all():
            print(f"  {c[2:]:<26}{'not run':>9}"); continue
        ret = v.mean() / base.mean() * 100 if base.mean() else np.nan
        print(f"  {c[2:]:<26}{v.mean():>+9.3f}{int((v > 0).sum()):>6}/9{ret:>21.0f}%")
    print("""
  READING THIS (A6). If a FROZEN arm retains most of the trained effect, the SAE's learning is
  not what produces H1, and the honest claim becomes "L_struct measures activation geometry"
  rather than "SAE features are more structurally local". That is weaker but defensible, and it
  is consistent with B1/B2/B4 (which use no SAE at all). If the frozen arms collapse toward 0,
  the learning matters and the project can say so with evidence instead of assumption.""")


# ----------------------------------------------------------------------------------
#                       R9 — multiplicity across the 9 depths
# ----------------------------------------------------------------------------------

def multiplicity(csv="outputs_robustness/bootstrap_h1_full_bylevel_minact0.csv",
                 traces="outputs_robustness/bootstrap_h1_traces_bylevel_minact0.npz",
                 level="fold", alpha=0.05):
    p = Path(csv)
    if not p.exists():
        print(f"  [skip] {csv} not found")
        return
    df = pd.read_csv(p)
    df = df[df.cluster_level == level].copy()
    if df.empty:
        print(f"  [skip] no rows at cluster_level={level}")
        return
    # Two-sided bootstrap p-value from the sign fraction: p = 2*min(f, 1-f), floored at 1/B.
    f = df["frac_pos"].values
    B = 1000
    pv = np.clip(2 * np.minimum(f, 1 - f), 1.0 / B, 1.0)
    rej, q = benjamini_hochberg(pv, alpha)
    df["p_boot"], df["q_BH"], df["sig_FDR"] = pv, q, rej

    print("=" * 88)
    print(f"R9  Multiplicity across the 9 depths (cluster_level={level}, BH-FDR alpha={alpha})")
    print("=" * 88)
    print(df[["rel_depth", "d_point", "ci_low", "ci_high", "frac_pos", "p_boot", "q_BH",
              "sig_FDR"]].to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}"))
    print(f"  survive BH-FDR: {int(rej.sum())}/{len(rej)}")

    tp = Path(traces)
    if not tp.exists():
        print(f"\n  [omnibus skipped] no traces at {traces} — rerun compute_h1_bootstrap.py")
        return
    z = np.load(tp)
    # Trace keys are '{split}_{level}_{depth}', e.g. full_fold_50 / val_protein_13. Selecting on
    # a bare substring would pool the FULL-set and VAL-set traces into one 18-column matrix and
    # silently report their average as the omnibus. Anchor on an exact '{split}_{level}_' prefix.
    print("\n  OMNIBUS (mean effect pooled across depths, within the SAME cluster resamples):")
    for split in ("full", "val"):
        pre = f"{split}_{level}_"
        keys = sorted([k for k in z.files if k.startswith(pre)],
                      key=lambda s: int(s.rsplit("_", 1)[1]))
        if len(keys) < 2:
            print(f"    [{split}] not enough traces ({len(keys)})")
            continue
        mat = np.column_stack([z[k] for k in keys])
        pooled = mat.mean(1)
        print(f"    [{split}, n={len(keys)} depths] mean d = {pooled.mean():+.4f}  95% CI "
              f"[{np.percentile(pooled,2.5):+.4f}, {np.percentile(pooled,97.5):+.4f}]  "
              f"frac>0 = {float((pooled>0).mean()):.3f}")
    print("    (one pre-registered global test; report ALONGSIDE the per-depth CIs, not instead)")


# ----------------------------------------------------------------------------------
#                     E5 — crosscoder rank-gap guard, across runs
# ----------------------------------------------------------------------------------

def crosscoder_partials(root="outputs_crosscoder"):
    rows = []
    for d in sorted(Path(root).glob("*/latent_diffing.csv")):
        lat = pd.read_csv(d)
        if "struct_delta" not in lat or "delta_norm" not in lat:
            continue
        raw = lat[["delta_norm", "struct_delta"]].corr(method="spearman").iloc[0, 1]
        covs = [c for c in ("activity", "firing_rate", "var_total") if c in lat]
        pr = partial_spearman(lat["delta_norm"].values, lat["struct_delta"].values,
                              lat[covs].values) if covs else np.nan
        meta = json.loads((d.parent / "META.json").read_text())
        rows.append({"run": d.parent.name, "freeze": meta.get("freeze", "none"),
                     "rho_raw": raw, "rho_partial": pr,
                     "retained_pct": pr / raw * 100 if raw else np.nan,
                     "val_EV_a": meta.get("val_EV_a"), "val_EV_b": meta.get("val_EV_b"),
                     "dn_median": meta.get("delta_norm_median")})
    if not rows:
        print("  [skip] no crosscoder runs with L_struct yet")
        return None
    df = pd.DataFrame(rows)
    print("=" * 88)
    print("E5  Crosscoder: raw vs PARTIAL Spearman(Delta_norm, L_struct)  [rank-gap guard, R5]")
    print("=" * 88)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("""
  READING THIS. rho_partial controls for latent activity, firing rate and total explained
  variance. If rho collapses toward 0 once those are removed, the shared/specific gradient was
  driven by one arm being easier to reconstruct (the effective-rank gap that killed Appendix O),
  not by the objective. A frozen-encoder run with a non-zero rho is the same warning by a
  different route: random features should not produce a structural gradient.""")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "a6", "a7", "r9", "e5"])
    ap.add_argument("--outlier-root", default="outputs_outlier")
    ap.add_argument("--crosscoder-root", default="outputs_crosscoder")
    ap.add_argument("--level", default="fold", choices=["fold", "protein"])
    ap.add_argument("--bootstrap-stem", default="bootstrap_h1",
                    help="stem of the compute_h1_bootstrap.py outputs to read for R9. "
                         "The default is the ESM-2 / RITA pair; the controlled pair this "
                         "paper reports is written under --preset ctrl_esmc, whose stem is "
                         "bootstrap_h1_ctrl_esmc. Passing the wrong stem makes R9 skip "
                         "rather than fail, which is how it went unnoticed.")
    ap.add_argument("--bootstrap-dir", default="outputs_robustness")
    args = ap.parse_args()

    if args.only in ("all", "a6", "a7"):
        df = compare_variants(args.outlier_root)
        if df is None:
            print("  [skip] no variant runs on disk yet (run run_sae_baselines.sh)")
        else:
            report_variants(df)
            Path("results_rigor").mkdir(exist_ok=True)
            df.to_csv("results_rigor/variant_comparison.csv", index=False)
        print()
    if args.only in ("all", "r9"):
        stem, bdir = args.bootstrap_stem, args.bootstrap_dir
        multiplicity(csv=f"{bdir}/{stem}_full_bylevel_minact0.csv",
                     traces=f"{bdir}/{stem}_traces_bylevel_minact0.npz",
                     level=args.level)
        print()
    if args.only in ("all", "e5"):
        df = crosscoder_partials(args.crosscoder_root)
        if df is not None:
            Path("results_rigor").mkdir(exist_ok=True)
            df.to_csv("results_rigor/crosscoder_partials.csv", index=False)


if __name__ == "__main__":
    main()
