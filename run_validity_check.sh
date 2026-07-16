#!/usr/bin/env bash
# run_validity_check.sh — the SAE-validity check on the trained checkpoints.
#
# Run this if you STILL HAVE ~/own_sae_data (the checkpoints). It re-extracts raw
# activations from each model and reports, per depth:
#   - effective rank of the activations (SAE-free)     <- the key number
#   - SAE val_EV (>=0.99 == degenerate basis)
# Output is a single small JSON (results_rank_ev/summary.json) to send back.
#
# Takes ~15-40 min on the GPU (a forward pass + one SAE per depth per model).
#   bash run_validity_check.sh
#   NO_EV=1 bash run_validity_check.sh     # rank only, no SAE, ~5 min
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUT="${OUT:-results_rank_ev/summary.json}"
EVFLAG=""; [ "${NO_EV:-0}" = "1" ] && EVFLAG="--no-ev"

declare -a NAMES=(clm mlm_token mlm_pred)
declare -a DIRS=(ckpt_clm_s42 ckpt_mlm_s42_token ckpt_mlm_s42_pred)

any=0
for i in "${!NAMES[@]}"; do
  ck="$DATA/${DIRS[$i]}/model_final.pt"
  if [ ! -f "$ck" ]; then echo "[skip] ${NAMES[$i]}: no $ck"; continue; fi
  any=1
  echo "=== ${NAMES[$i]}  ($ck)  $(date) ==="
  $PY -u measure_rank_ev.py --ckpt "$ck" --name "${NAMES[$i]}" \
      --eval-set eval_set --out "$OUT" $EVFLAG || echo "!! ${NAMES[$i]} failed"
done
[ "$any" = "0" ] && { echo "No checkpoints found under $DATA — nothing to do."; exit 1; }
echo; echo "DONE -> $OUT   (send this one small file back)"
