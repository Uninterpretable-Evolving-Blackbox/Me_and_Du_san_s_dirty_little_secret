#!/usr/bin/env bash
# run_500tpp.sh — does the L_struct result survive 32x more training?
#
# THE QUESTION. Every main result sits at 15.7 tokens/param. The token ablation shows the
# two arms move in OPPOSITE directions with more training, in exactly the layers the main
# results use (annealed-to-annealed, participation ratio):
#
#            L11          L14          L18
#   MLM   12.8 -> 12.8  14.3 -> 14.7  23.4 -> 35.0     (flat to rising)
#   CLM   22.6 ->  6.9  24.9 ->  9.2  32.8 -> 12.9     (collapses ~3x)
#
# so the CLM > MLM ordering at 15.7 becomes MLM > CLM at 500. If L_struct's MLM-CLM gap
# tracks that, the headline effect is a property of the training budget, not the objective.
#
# WHAT THIS RUNS. Nothing new is trained. The 21B-token (500 tok/param) checkpoints already
# exist from run_token_ablation.sh, annealed, seed 42, same corpus and recipe as stage 1 --
# only the budget differs. This just extracts + fits an SAE + runs cpu_stage on them, with
# the project's settings (sae_seed 42, expansion 8, k 256, n-shuffles 5).
#
# The ablation's MLM arm is token-matched (run_token_ablation.sh passes --objective mlm with
# no --match-predictions), so it pairs with ckpt_mlm_s42_token, not the pred arm.
#
# Single process throughout: eval_ctrl_plm and cpu_stage are each internally parallel.
set -u
cd "$(dirname "$0")"
PY="${PY:-$PWD/.venv/bin/python}"
ABL="$HOME/own_sae_data/uniref50_pilot/token_ablation"
OUT=outputs_ctrl_500tpp
LAYERS="11 14 18"
N_SHUFFLES=5

echo "########## 500 tok/param controls | $(date) ##########"
ok=0; fail=0
for spec in "ckpt_mlm_s42_token:$ABL/ckpt_mlm_s42/model_final.pt" \
            "ckpt_clm_s42:$ABL/ckpt_clm_s42/model_final.pt"; do
  name="${spec%%:*}"; ck="${spec#*:}"
  if [ ! -f "$ck" ]; then echo "!! missing $ck"; fail=$((fail+1)); continue; fi
  for L in $LAYERS; do
    LD="$OUT/$name/layer_$L"
    if [ -f "$LD/struct_seq_metrics.csv" ]; then echo "[$name L$L] done - skip"; ok=$((ok+1)); continue; fi
    echo "=== [$name L$L] extract + SAE $(date) ==="
    if ! "$PY" -u eval_ctrl_plm.py --ckpt "$ck" --name "$name" --layer "$L" \
         --out-root "$OUT" --eval-set eval_set \
         --sae-seed 42 --expansion 8 --k-sparse 256; then
      echo "!! eval failed $name L$L"; fail=$((fail+1)); continue
    fi
    echo "=== [$name L$L] cpu_stage $(date) ==="
    if "$PY" -u cpu_stage.py --layer-dir "$LD" --model-type residue \
         --n-shuffles "$N_SHUFFLES" \
         --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
         --fasta-path cache/scope_40.fa; then
      ok=$((ok+1))
    else
      echo "!! cpu_stage failed $name L$L"; fail=$((fail+1))
    fi
  done
done

echo
echo "########## DONE $(date)  ok=$ok fail=$fail ##########"
"$PY" - <<'PY'
import csv, os
import numpy as np
def ms(root, arm, L):
    f = "%s/%s/layer_%d/struct_seq_metrics.csv" % (root, arm, L)
    if not os.path.exists(f):
        return None
    return float(np.mean([float(r["struct_delta"]) for r in csv.DictReader(open(f))]))
print()
print("mean struct_delta -- 15.7 tok/param (paper) vs 500 tok/param (32x)")
print()
print("  layer |   MLM 15.7   MLM 500 |   CLM 15.7   CLM 500 |  gap 15.7   gap 500")
for L in (11, 14, 18):
    a0, a1 = ms("outputs_ctrl", "ckpt_mlm_s42_token", L), ms("outputs_ctrl_500tpp", "ckpt_mlm_s42_token", L)
    b0, b1 = ms("outputs_ctrl", "ckpt_clm_s42", L), ms("outputs_ctrl_500tpp", "ckpt_clm_s42", L)
    if None in (a0, a1, b0, b1):
        print("  L%-4d  incomplete" % L); continue
    print("  L%-4d | %+9.5f %+9.5f | %+9.5f %+9.5f | %+9.5f %+9.5f"
          % (L, a0, a1, b0, b1, a0 - b0, a1 - b1))
print()
print("  if gap 500 keeps sign and rough magnitude -> the effect is budget-stable")
print("  if it shrinks to ~0 or flips -> the headline result is a property of the budget")
PY
