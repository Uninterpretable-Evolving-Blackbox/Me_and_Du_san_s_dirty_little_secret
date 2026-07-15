#!/usr/bin/env bash
# run_multiseed_cuda.sh — controlled MLM-vs-CLM, multi-seed, on a CUDA GPU.
#
# Trains one MLM and one CLM model per seed. Within a seed both objectives share
# the SAME init, SAME corpus and SAME batch order; ONLY the objective differs.
# Across seeds only --seed changes. That is the whole experiment.
#
# Every other hyperparameter below is pinned to the original run. DO NOT tune them
# for speed (see README) — changing batch size or LR makes the seeds incomparable.
# Speed should come from the GPU, not from a different recipe.
#
# Self-healing: resumes from the newest checkpoint after a crash. Idempotent: an
# objective/seed whose model_final.pt exists is skipped, so re-running is safe.
#
#   bash run_multiseed_cuda.sh                 # seeds 42 43 44 (default)
#   SEEDS="42 43" bash run_multiseed_cuda.sh   # subset
set -u

cd "$(dirname "$0")"
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
SEEDS="${SEEDS:-42 43 44}"
MAXTRIES=5

# ---- pinned recipe: identical to the original seed-42 run. Do not edit. ----
TARGET=660e6        # token budget per model
BATCH=32            # NOT a speed knob - changes optimisation
SEQ=512
LR=6e-4
WARMUP=500
CKPT_EVERY=2500
VAL_EVERY=1000
# ---------------------------------------------------------------------------

if [ ! -f "$DATA/tokens.npy" ]; then
  echo "ERROR: corpus not found at $DATA/tokens.npy"
  echo "Run step 1 first:  $PY prep_controlled_corpus.py"
  exit 1
fi

$PY - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch cannot see a CUDA GPU. See README 'Install torch for Blackwell'.")
print(f"GPU OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}")
PY

latest_ckpt () { ls -t "$1"/model_step*.pt "$1"/model_partial.pt 2>/dev/null | head -1; }

train_one () {  # $1=objective  $2=seed
  local OBJ="$1" SEED="$2" D="$DATA/seed$2/ckpt_$1"
  mkdir -p "$D"
  if [ -f "$D/model_final.pt" ]; then echo "[seed $SEED/$OBJ] already done - skip"; return 0; fi
  for try in $(seq 1 $MAXTRIES); do
    local lc res=""; lc=$(latest_ckpt "$D"); [ -n "$lc" ] && res="--resume $lc"
    echo "=== [seed $SEED/$OBJ] try $try resume='${lc:-none}' $(date) ==="
    $PY -u train_ctrl_plm.py \
        --objective "$OBJ" --data-dir "$DATA" --out-dir "$D" \
        --seed "$SEED" \
        --target-tokens "$TARGET" --batch-size "$BATCH" --seq-len "$SEQ" \
        --lr "$LR" --warmup "$WARMUP" \
        --ckpt-every "$CKPT_EVERY" --val-every "$VAL_EVERY" $res \
      && return 0
    echo "[seed $SEED/$OBJ] try $try FAILED $(date) - resuming"; sleep 15
  done
  echo "[seed $SEED/$OBJ] GAVE UP after $MAXTRIES tries"; return 1
}

echo "########## controlled MLM-vs-CLM | seeds: $SEEDS | $(date) ##########"
for s in $SEEDS; do
  for obj in mlm clm; do
    train_one "$obj" "$s" || echo "!! seed $s/$obj did not finish"
  done
done

echo
echo "########## DONE $(date) ##########"
echo "Checkpoints:"
for s in $SEEDS; do
  for obj in mlm clm; do
    f="$DATA/seed$s/ckpt_$obj/model_final.pt"
    [ -f "$f" ] && echo "  OK   $f" || echo "  MISS $f"
  done
done
echo
echo "Send back every model_final.pt (~400MB each) plus the console log."
