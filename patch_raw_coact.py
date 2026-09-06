"""Fix the checkpoint-key bug in experiment_raw_coactivation.py.

train_ctrl_plm.py writes the model config under "cfg". eval_ctrl_plm.py:164 and
eval_ctrl_saefree.py:270 both read "cfg". experiment_raw_coactivation.py (new in
0cab269, never run against these checkpoints) reads "config", so the lookup misses
and it falls through to CtrlESMC() -- which cannot work, because vocab_size is a
required positional argument. All four stage-10 cells died with the same TypeError.

Two sites, both in the same block: the model construction and the `causal` lookup.
The causal fallback happened to give the right answer via "clm" in name, but it was
reading an empty dict to get there.
"""
import ast
import sys

P = "experiment_raw_coactivation.py"
src = open(P).read()

OLD_MODEL = '        model = CtrlESMC(**ck["config"]) if "config" in ck else CtrlESMC()'
NEW_MODEL = (
    '        # train_ctrl_plm.py stores the model config under "cfg", not "config";\n'
    '        # eval_ctrl_plm.py:164 and eval_ctrl_saefree.py:270 both read "cfg".\n'
    '        # The old CtrlESMC() fallback cannot work -- vocab_size is required.\n'
    '        cfg = ck["cfg"] if "cfg" in ck else ck.get("config", {})\n'
    '        model = CtrlESMC(**cfg)'
)
OLD_CAUSAL = '        causal = bool(ck.get("config", {}).get("causal", "clm" in args.name))'
NEW_CAUSAL = '        causal = bool(cfg.get("causal", "clm" in args.name))'

for name, old in (("model", OLD_MODEL), ("causal", OLD_CAUSAL)):
    if old not in src:
        sys.exit("FAIL: %s line not found verbatim -- aborting without writing" % name)

src = src.replace(OLD_MODEL, NEW_MODEL).replace(OLD_CAUSAL, NEW_CAUSAL)
ast.parse(src)
open(P, "w").write(src)
print("patched both sites; syntax OK")
