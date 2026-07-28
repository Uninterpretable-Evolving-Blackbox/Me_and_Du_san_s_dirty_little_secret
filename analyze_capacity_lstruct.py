#!/usr/bin/env python3
"""
analyze_capacity_lstruct.py — does L_struct actually depend on SAE reconstruction quality?

THE UNTESTED ASSUMPTION
  Everything we have worried about regarding "instrument quality" rests on this chain:

      low val_EV -> the SAE discards variance -> features are a lossier representation
                 -> co-activation statistics are attenuated -> L_struct is depressed

  Not one link has ever been measured. The k x expansion sweep produced val_EV, ev_gap,
  dead counts and participation ratios across 24 configurations — and no L_struct at all.
  So the Spearman(EV gap, d) = +0.66 correlation across depths has never been checked
  against the far more direct question: within a single arm, holding the model fixed and
  varying ONLY the dictionary, does L_struct move with val_EV?

WHY THIS IS THE RIGHT TEST
  Across depths, val_EV and L_struct both vary with depth, so a correlation between them
  is expected even if neither causes the other. Within an arm at a fixed layer, depth is
  held constant and the only thing changing is SAE capacity. That isolates the link.

  The CLM arm is the powerful case: capacity settings drive its val_EV from ~0.81 down to
  roughly ZERO (a dictionary that reconstructs essentially nothing). If L_struct is flat
  across that range, reconstruction quality is not what L_struct measures, the EV-gap
  objection dissolves, and a defensive section of the paper can be deleted.

WHAT EACH OUTCOME MEANS
  flat  -> the EV gap is a red herring. Report this figure and stop apologising for it.
  steep -> the confound is real, and now it is quantified rather than assumed, which also
           tells you how much of the observed effect it could account for.

  Either way this replaces an assumption with a measurement, which is the whole point.

Input: cells written by run_all_pending.sh stage 6, laid out as
       outputs_ctrl_cap/{arm}_k{K}e{E}/layer_{L}/{struct_seq_metrics.csv,META.json}

Usage
  python analyze_capacity_lstruct.py --root outputs_ctrl_cap --layer 14
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

CELL = re.compile(r"^(?P<arm>.+)_k(?P<k>\d+)e(?P<e>\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_ctrl_cap")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--out", default="results_rigor/capacity_vs_lstruct.csv")
    args = ap.parse_args()

    rows = []
    for d in sorted(Path(args.root).glob("*")):
        m = CELL.match(d.name)
        if not m:
            continue
        cell = d / f"layer_{args.layer}"
        csv, meta = cell / "struct_seq_metrics.csv", cell / "META.json"
        if not (csv.exists() and meta.exists()):
            continue
        sd = pd.read_csv(csv)["struct_delta"].to_numpy(np.float64)
        mt = json.loads(meta.read_text())
        rows.append(dict(arm=m.group("arm"), k=int(m.group("k")), expansion=int(m.group("e")),
                         val_EV=mt.get("val_EV"), hidden=mt.get("sae_hidden_dim"),
                         mean_lstruct=float(sd.mean()), median_lstruct=float(np.median(sd)),
                         n_features=len(sd)))
    if not rows:
        raise SystemExit(f"no cells under {args.root} at layer {args.layer}")

    df = pd.DataFrame(rows).sort_values(["arm", "val_EV"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("=" * 92)
    print(f"DOES L_struct TRACK val_EV?  layer {args.layer}, model fixed, dictionary varied")
    print("=" * 92)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print()
    for arm, g in df.groupby("arm"):
        if len(g) < 3:
            print(f"  {arm}: only {len(g)} configs — need >=3")
            continue
        ev, ls = g.val_EV.to_numpy(float), g.mean_lstruct.to_numpy(float)
        rho, p_s = spearmanr(ev, ls)
        r, p_p = pearsonr(ev, ls)
        slope = np.polyfit(ev, ls, 1)[0]
        span_ev, span_ls = ev.max() - ev.min(), ls.max() - ls.min()
        print(f"  {arm}")
        print(f"    val_EV spans {ev.min():+.4f} -> {ev.max():+.4f}  (range {span_ev:.4f})")
        print(f"    L_struct spans {ls.min():+.4f} -> {ls.max():+.4f}  (range {span_ls:.4f})")
        print(f"    Spearman rho = {rho:+.3f} (p={p_s:.3f})   Pearson r = {r:+.3f} (p={p_p:.3f})")
        print(f"    slope dL_struct/dval_EV = {slope:+.4f}")
        print(f"    -> a 0.30 val_EV gap (the observed MLM-CLM gap) would move L_struct by "
              f"{abs(slope) * 0.30:+.4f}")

    # The decisive comparison: how does that predicted movement compare with the effect?
    print()
    print("  READING THIS. The observed MLM-CLM val_EV gap at this layer is ~0.30. If the")
    print("  predicted L_struct movement above is small next to the actual MLM-CLM difference")
    print("  in mean L_struct, then reconstruction quality cannot account for the effect and")
    print("  the EV-gap objection is answered with a measurement instead of an argument.")
    print("  If it is comparable, the confound is real and its size is now known.")
    print(f"\n  wrote -> {args.out}")


if __name__ == "__main__":
    main()
