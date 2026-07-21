#!/usr/bin/env bash
# run_token_ablation.sh — how does representation rank change with TRAINING LENGTH?
#
# WHY THIS EXPERIMENT
#   Our 42M models were trained on 660M tokens = ~16 tokens/parameter. Real protein
#   LMs (ESM-C, ESM-2) are trained at THOUSANDS of tokens/parameter - two to three
#   orders of magnitude more. So when our small models look different from the real
#   ones, we cannot tell whether that is about model SIZE or simply about how long
#   they were trained. This varies training length while holding size fixed.
#
# DESIGN
#   ONE long run per objective, with checkpoints saved at token milestones. The whole
#   rank-vs-tokens curve therefore costs the same as its single longest run - no need
#   for separate runs per budget. Both arms are identical except MLM vs CLM.
#
#   Milestones (42M params): 0.66B(~16/param, matches the run you already did),
#   2.1B(50), 4.2B(100), 10.5B(250), 21B(500).
#
# COST (measured 233k tok/s on the RTX PRO 6000 for this model size)
#   21B tokens ~= 25 h per arm, so ~50 h for both. Fully resumable - interrupt freely.
#   MAX_TOKENS=4.2e9 bash run_token_ablation.sh   # ~10 h total, still a 6x span
#
# CAVEAT (documented on purpose)
#   The LR cosine spans the FULL budget, so intermediate milestones are mid-schedule,
#   not fully-annealed models. That is fine for this question - we are asking how rank
#   EVOLVES, and both arms share an identical schedule, so the comparison is fair.
#
#   bash run_token_ablation.sh
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUTROOT="${OUTROOT:-$DATA/token_ablation}"
MAX_TOKENS="${MAX_TOKENS:-21e9}"
MILESTONES="${MILESTONES:-0.66e9,2.1e9,4.2e9,10.5e9,21e9}"
SEED="${SEED:-42}"
MAXTRIES=5

# ---- pinned recipe: identical to your existing runs. Only training LENGTH varies. ----
BATCH=32; SEQ=512; LR=6e-4; WARMUP=500
D_MODEL=320; N_HEADS=5; N_LAYERS=30        # ESM-C anchor: head_dim 64, depth 30
CKPT_EVERY=5000                            # rolling resume checkpoint (overwrites)
VAL_EVERY=2000
LAYERS="0,4,7,11,14,18,22,26,29"           # the 9 paper-matched relative depths
# --------------------------------------------------------------------------------------

[ -f "$DATA/tokens.npy" ] || { echo "ERROR: no corpus at $DATA. Run: $PY prep_controlled_corpus.py"; exit 1; }
$PY - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available(): sys.exit("ERROR: no CUDA GPU visible.")
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
import esm  # noqa: F401
PY

echo "########## TOKEN ABLATION | max=${MAX_TOKENS} tok | milestones=${MILESTONES} | $(date) ##########"

train_arm () {
  local OBJ="$1" D="$OUTROOT/ckpt_${OBJ}_s${SEED}"
  mkdir -p "$D"
  if [ -f "$D/model_final.pt" ]; then echo "[$OBJ] already complete - skip"; return 0; fi
  for try in $(seq 1 $MAXTRIES); do
    local res=""
    [ -f "$D/model_resume.pt" ] && res="--resume $D/model_resume.pt"
    echo "=== [$OBJ] try $try ${res:+(resuming)} $(date) ==="
    $PY -u train_ctrl_plm.py --objective "$OBJ" --data-dir "$DATA" --out-dir "$D" \
        --seed "$SEED" --target-tokens "$MAX_TOKENS" \
        --ckpt-at-tokens "$MILESTONES" --rolling-resume \
        --batch-size "$BATCH" --seq-len "$SEQ" --lr "$LR" --warmup "$WARMUP" \
        --ckpt-every "$CKPT_EVERY" --val-every "$VAL_EVERY" \
        --d-model "$D_MODEL" --n-heads "$N_HEADS" --n-layers "$N_LAYERS" $res \
      && return 0
    echo "[$OBJ] try $try FAILED - resuming"; sleep 15
  done
  echo "[$OBJ] GAVE UP after $MAXTRIES tries"; return 1
}

# ---- 1. train both arms (long runs, milestones saved along the way) ----
for obj in mlm clm; do train_arm "$obj" || echo "!! $obj did not finish"; done

# ---- 2. measure rank at every milestone (SAE-free, ~2 min each) ----
echo; echo "########## MEASURING RANK vs TOKENS $(date) ##########"
OUT="results_token_ablation/summary.json"
for obj in mlm clm; do
  for ck in "$OUTROOT/ckpt_${obj}_s${SEED}"/model_tok*M.pt; do
    [ -f "$ck" ] || continue
    tag="${obj}_$(basename "$ck" .pt | sed 's/model_//')"
    echo "=== $tag ==="
    $PY -u measure_rank_ev.py --ckpt "$ck" --name "$tag" --eval-set eval_set \
        --layers "$LAYERS" --no-ev --out "$OUT" || echo "!! $tag failed"
  done
done

echo; echo "########## DONE $(date) ##########"
echo "Send back this one file:  $OUT"
echo "(and train.log, so we can check both arms trained comparably)"
