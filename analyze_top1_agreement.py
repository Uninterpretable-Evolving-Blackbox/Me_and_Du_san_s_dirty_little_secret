#!/usr/bin/env python3
"""
analyze_top1_agreement.py — does the composition check survive its own robustness measure?

WHY
---
Section 3's amino-acid composition check reports Spearman rho between a
background-corrected selectivity score S and L_struct, finds it weak and
negative, and concludes that composition does not drive the metric. That
conclusion rests on one operationalisation of "selective".

experiment_aa_selectivity.py already computes a second, cruder one alongside it
-- `top1_share`, the fraction of background-corrected mass on the single
most-preferred residue type -- and its own docstring states the decision rule:

    "If the two disagree the result is fragile and should not be reported."

The numbers exist. This script applies that rule and prints the verdict, so the
outcome does not depend on who reads the CSV.

WHAT COUNTS AS AGREEMENT
------------------------
Fixed before looking at the data, and printed on every run:

  1. SIGN     both correlations point the same way, or both are within the
              null band |rho| < --null-band (default 0.05), where sign is
              not interpretable anyway;
  2. MAGNITUDE  both are weak: |rho| < --weak (default 0.20). The paper's own
              largest value is 0.122, so 0.20 leaves headroom without letting
              a moderate correlation through;
  3. VERDICT  both support the same conclusion about the paper's claim, which
              is "composition does not drive L_struct".

Failing any of these on any cell means the composition bullet in Section 3 is
not robust to the choice of selectivity measure. In that case the honest move
is to report both correlations and drop the check count from six to five.

USAGE
    python analyze_top1_agreement.py
    python analyze_top1_agreement.py --summary results_rigor/aa_selectivity.csv
    python analyze_top1_agreement.py --per-feature results_rigor/aa_selectivity_per_feature.csv

Exit status is 0 whether the check agrees or disagrees -- a disagreement is a
finding, not a crash. It is non-zero only if the inputs are missing or malformed,
so a driver script cannot mistake "never ran" for "agreed".
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_SUMMARY = ("cell", "rho_selectivity_struct", "rho_top1_struct")


def _spearman(x, y):
    from scipy.stats import spearmanr
    g = np.isfinite(x) & np.isfinite(y)
    if g.sum() < 3:
        return float("nan"), float("nan"), int(g.sum())
    r, p = spearmanr(x[g], y[g])
    return float(r), float(p), int(g.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results_rigor/aa_selectivity.csv")
    ap.add_argument("--per-feature", default="results_rigor/aa_selectivity_per_feature.csv")
    ap.add_argument("--weak", type=float, default=0.20,
                    help="|rho| below this counts as 'weak' (default 0.20)")
    ap.add_argument("--null-band", type=float, default=0.05,
                    help="|rho| below this has uninterpretable sign (default 0.05)")
    ap.add_argument("--out", default="results_rigor/top1_agreement.csv")
    args = ap.parse_args()

    summary_p, perfeat_p = Path(args.summary), Path(args.per_feature)
    if not summary_p.exists():
        print(f"ERROR: no {summary_p}", file=sys.stderr)
        print("  Generate it with:", file=sys.stderr)
        print("    python experiment_aa_selectivity.py --root <outputs_root> \\",
              file=sys.stderr)
        print("      --cells <model>:<layer>,... --out results_rigor/aa_selectivity.csv",
              file=sys.stderr)
        return 2

    df = pd.read_csv(summary_p)
    missing = [c for c in REQUIRED_SUMMARY if c not in df.columns]
    if missing:
        print(f"ERROR: {summary_p} is missing column(s) {missing}", file=sys.stderr)
        print(f"  columns present: {list(df.columns)}", file=sys.stderr)
        return 2
    if len(df) == 0:
        print(f"ERROR: {summary_p} has no rows", file=sys.stderr)
        return 2

    print("=" * 74)
    print("top1_share agreement check")
    print(f"  summary:     {summary_p}")
    print(f"  per-feature: {perfeat_p if perfeat_p.exists() else '(absent)'}")
    print(f"  rule: same sign (or both |rho| < {args.null_band}) "
          f"AND both |rho| < {args.weak}")
    print("=" * 74)

    # direct agreement between the two measures, if the per-feature file is there
    direct = {}
    if perfeat_p.exists():
        pf = pd.read_csv(perfeat_p)
        need = {"cell", "selectivity", "top1_share"}
        if need.issubset(pf.columns):
            for cell, g in pf.groupby("cell"):
                r, p, n = _spearman(g["selectivity"].to_numpy(np.float64),
                                    g["top1_share"].to_numpy(np.float64))
                direct[str(cell)] = (r, p, n)
        else:
            print(f"  note: {perfeat_p} lacks {sorted(need - set(pf.columns))}; "
                  "skipping the direct measure-agreement column")

    rows, disagreements = [], []
    print(f"\n{'cell':<24}{'rho_S':>9}{'rho_top1':>10}{'rho(S,top1)':>13}  verdict")
    print("-" * 74)
    for _, r in df.iterrows():
        cell = str(r["cell"])
        rs = float(r["rho_selectivity_struct"])
        rt = float(r["rho_top1_struct"])
        dr, dp, dn = direct.get(cell, (float("nan"),) * 3)

        both_null = abs(rs) < args.null_band and abs(rt) < args.null_band
        sign_ok = both_null or (np.sign(rs) == np.sign(rt))
        weak_ok = abs(rs) < args.weak and abs(rt) < args.weak
        agree = bool(sign_ok and weak_ok)

        why = []
        if not sign_ok:
            why.append("opposite sign")
        if not weak_ok:
            why.append(f"|rho| >= {args.weak}")
        verdict = "AGREE" if agree else "DISAGREE (" + ", ".join(why) + ")"
        if not agree:
            disagreements.append(cell)

        dstr = f"{dr:+.3f}" if np.isfinite(dr) else "n/a"
        print(f"{cell:<24}{rs:>+9.3f}{rt:>+10.3f}{dstr:>13}  {verdict}")
        rows.append(dict(cell=cell, rho_selectivity_struct=rs, rho_top1_struct=rt,
                         rho_selectivity_vs_top1=dr, p_selectivity_vs_top1=dp,
                         n_features_direct=dn, sign_ok=sign_ok, weak_ok=weak_ok,
                         agree=agree))

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("-" * 74)
    if not disagreements:
        print(f"VERDICT: AGREE on all {len(out)} cell(s).")
        print("  Section 3's composition check is robust to the selectivity measure.")
        print("  No change to the paper. The check count stays at six.")
    else:
        print(f"VERDICT: DISAGREE on {len(disagreements)} of {len(out)} cell(s): "
              f"{', '.join(disagreements)}")
        print("  Section 3's composition bullet is NOT robust to the choice of")
        print("  selectivity measure. Report both correlations, and drop the")
        print("  check count from six to five in the abstract, Section 1,")
        print("  Section 3 and Section 5 Scope.")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
