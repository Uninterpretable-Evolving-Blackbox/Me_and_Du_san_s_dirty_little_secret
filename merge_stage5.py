"""Merge the 18 stage-5 chunk CSVs into what the serial run would have written.

The chunk column is `arm` (not `model` — that was my merge script's bug). Everything
below the concat is character-for-character the tail of eval_ctrl_saefree.py, so the
two output CSVs are what a single serial invocation would have produced.
"""
import glob
import sys

import numpy as np
import pandas as pd

FINAL = "results_ctrl_saefree"

parts = sorted(glob.glob(FINAL + "_part_*/saefree_by_arm.csv"))
if not parts:
    sys.exit("no part CSVs to merge")

df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset=["arm", "layer"], keep="first").reset_index(drop=True)
if before != len(df):
    print("dropped %d duplicate (arm,layer) rows" % (before - len(df)))
df = df.sort_values(["arm", "layer"]).reset_index(drop=True)
df.to_csv(FINAL + "/saefree_by_arm.csv", index=False)

# identical expression to eval_ctrl_saefree.py's tail
metrics = [c for c in df.columns if c.startswith("probe_") or c == "contact_p_at_L5"]
mlm = df[~df.causal].groupby("rel_depth")[metrics].mean()
clm = df[df.causal].groupby("rel_depth")[metrics].mean()
common = [d for d in mlm.index if d in clm.index]
diff = (mlm.loc[common] - clm.loc[common]).reset_index()
diff.to_csv(FINAL + "/saefree_mlm_minus_clm.csv", index=False)

print("merged %d part files -> %d rows, %d arms, %d depths"
      % (len(parts), len(df), df.arm.nunique(), df.layer.nunique()))
print("=" * 88)
print("SAE-FREE:  MLM - CLM  (positive = masked better).  No SAE used anywhere.")
print("=" * 88)
order = ["0%", "13%", "25%", "38%", "50%", "63%", "75%", "88%", "100%"]
show = diff.set_index("rel_depth").reindex([d for d in order if d in set(diff.rel_depth)])
print(show.to_string(float_format=lambda v: "%+.4f" % v))
for m in metrics:
    if m in diff.columns:
        v = diff[m].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            print("  %-24s mean %+.4f  MLM>CLM at %d/%d"
                  % (m, v.mean(), int((v > 0).sum()), v.size))
