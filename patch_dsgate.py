"""Make a copy of experiment_interplm_metric.py whose 0.6 gate is dataset-wide.

WHY. InterPLM normalises each feature by its maximum ACROSS THE DATASET before applying
the > 0.6 threshold. From their own interplm/sae/normalize.py, which stores an
`activation_rescale_factor` buffer applied post-ReLU at inference:

    "ensuring that the maximum activation value for each feature is 1 across the
     provided dataset"

experiment_interplm_metric.py gates on each PROTEIN's own maximum instead:

    a = a_all[lo:hi]                       # one protein's residues
    if (a > gate * a.max() ...).sum() < 1: # a.max() is this protein's max
        continue

That condition can never fire. The line above it already establishes a.max() > 0, and
the argmax element always satisfies a > 0.6 * a.max(). Measured over 120 sampled
features on ckpt_clm_s42/layer_14: the current gate admits a median of 1485 of 1500
proteins per feature and 120/120 features clear the >= 25-protein rule. With the
dataset-wide max it is 98 proteins and 87/120 features -- a 15x change in the analysed
set, which feeds straight into mean_d_struct and n_significant.

Only the gate changes. Everything else is byte-identical to Wei's file, so the two runs
are a clean A/B on this one decision.
"""
import ast
import shutil
import sys

SRC = "experiment_interplm_metric.py"
DST = "experiment_interplm_metric_dsgate.py"

shutil.copy(SRC, DST)
src = open(DST).read()

ANCHOR = "        a_all = np.asarray(Z[:, f], dtype=np.float32)"
ANCHOR_NEW = (
    "        a_all = np.asarray(Z[:, f], dtype=np.float32)\n"
    "        # dataset-wide per-feature maximum, matching InterPLM's normalisation\n"
    "        _gmax = float(a_all.max())"
)
GATE = "            if (a > gate * a.max() if gate <= 1 else a > gate).sum() < 1:"
GATE_NEW = "            if (a > gate * _gmax if gate <= 1 else a > gate).sum() < 1:"

for name, old in (("anchor", ANCHOR), ("gate", GATE)):
    if old not in src:
        sys.exit("FAIL: %s not found verbatim in %s" % (name, SRC))

src = src.replace(ANCHOR, ANCHOR_NEW).replace(GATE, GATE_NEW)
ast.parse(src)
open(DST, "w").write(src)
print("wrote %s (gate is now dataset-wide); syntax OK" % DST)
