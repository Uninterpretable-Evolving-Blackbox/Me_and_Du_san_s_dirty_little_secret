#!/usr/bin/env bash
# run_ppl500.sh — perplexity of the 500 tok/param models.
#
# WHY. The reviewer asked whether the models reach low perplexity, on the reasoning that
# if they do, structure is likely encoded, so a null on co-activation indicts the metric
# rather than the models. We measured perplexity only at 15.7 tok/param (real CLM 16.430
# causal, real MLM 15.678 pseudo-masked), which is not obviously "low" for a 42M model
# and leaves "the models simply did not learn much structure" alive as an alternative.
#
# The 21B-token (500 tok/param) checkpoints are 32x better trained and have never had
# perplexity measured. If they come in clearly lower, the reviewer's premise is satisfied
# on models where we have already shown the L_struct gap survives (and roughly doubles).
#
# One layer only: perplexity does not depend on --layers, and the single layer's
# raw-dimension struct_delta at 500 tok/param is a useful side product to compare with
# the 15.7 tok/param raw numbers from stage 10.
#
# CAVEAT TO CARRY INTO ANY WRITE-UP: the perplexity path in experiment_raw_coactivation.py
# is code I patched (attention_mask, and the mask token taken from meta["mask"]). The
# masked-arm number in particular depends on that mask-token choice. Original file is
# experiment_raw_coactivation.py.orig.
set -u
cd "$(dirname "$0")"
PY="${PY:-$PWD/.venv/bin/python}"
ABL="$HOME/own_sae_data/uniref50_pilot/token_ablation"
OUT=results_raw_coactivation_500tpp
mkdir -p "$OUT"

echo "########## perplexity @ 500 tok/param | $(date) ##########"
for spec in "ckpt_mlm_s42_token:$ABL/ckpt_mlm_s42/model_final.pt" \
            "ckpt_clm_s42:$ABL/ckpt_clm_s42/model_final.pt"; do
  name="${spec%%:*}"; ck="${spec#*:}"
  out="$OUT/${name}.csv"
  if [ -f "$out" ]; then echo "[$name] cached"; continue; fi
  [ -f "$ck" ] || { echo "!! missing $ck"; continue; }
  echo "=== [$name] $(date) ==="
  "$PY" -u experiment_raw_coactivation.py --ckpt "$ck" --name "$name" \
    --layers 14 --eval-set eval_set --n-shuffles 5 --out "$out" \
    || echo "!! failed $name"
done

echo
echo "########## DONE $(date) ##########"
echo "  --- perplexity, 15.7 vs 500 tok/param ---"
grep -ah "perplexity" logs_mustruns/s10_real_*.log 2>/dev/null | sed 's/^ */  15.7  /'
grep -ah "perplexity" "$OUT"/../*.log 2>/dev/null | true
"$PY" - <<'PY'
import glob, os
import pandas as pd
rows = []
for f in sorted(glob.glob("results_raw_coactivation_500tpp/*.csv")):
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        rows.append(dict(arm=os.path.basename(f)[:-4], layer=r["layer"],
                         perplexity=r.get("perplexity"), kind=r.get("ppl_kind"),
                         raw_struct_delta=r.get("mean_struct_delta")))
if rows:
    print()
    print("  500 tok/param:")
    print(pd.DataFrame(rows).to_string(index=False))
print()
print("  causal and pseudo-masked perplexities are DIFFERENT quantities -- compare each")
print("  arm against its own 15.7 tok/param value, never across arms.")
PY
