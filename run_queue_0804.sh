#!/usr/bin/env bash
# run_queue_0804.sh — three jobs, strictly serial, one process at a time.
#
#   A  feature labelling by top-activating residues   ~2 h   analysis only
#   B  shuffled corpus, seeds 43 and 44               ~8 h   4 training runs
#   C  shuffled corpus pair at 500 tok/param          ~50 h  2 training runs
#
# SERIAL BY DESIGN. Every step here is already internally parallel (joblib n_jobs=-1 for
# the analysis, full GPU for training). Running several at once is what took the box down
# on 2026-08-03: eight concurrent sweeps x 32 workers each exhausted 123 GB and filled
# swap. Nothing in this file backgrounds anything.
#
# WHY B DOES NOT CALL `STAGE=5 bash RUN_MUSTRUNS.sh`
#   That stage invokes train_ctrl_plm.py with neither --target-tokens nor --warmup, so it
#   takes the script defaults: 700e6 tokens and warmup 200. Every model in this project --
#   including the seed-42 shuffled pair those runs must be compared against -- was trained
#   at 660e6 (659,996,672 tokens, step 40282) with warmup 500, per run_full_ctrl.sh:45,49.
#   Running it as written would put seeds 43/44 on a 6% larger budget and a different
#   warmup, silently making the multi-seed comparison invalid -- which is the one thing
#   the stage exists to fix. The recipe below matches run_full_ctrl.sh exactly.
set -u
cd "$(dirname "$0")"
PY="${PY:-$PWD/.venv/bin/python}"
LOG=queue_0804.log

# ---- recipe, copied from run_full_ctrl.sh lines 45-52 ----
TARGET=660e6; BATCH=32; SEQ=512; LR=6e-4; WARMUP=500
CKPT_EVERY=2500; VAL_EVERY=1000
D_MODEL=320; N_HEADS=5; N_LAYERS=30
SHUF="$HOME/own_sae_data/uniref50_pilot_shuf"
ABL_SHUF="$HOME/own_sae_data/uniref50_pilot_shuf_500tpp"

say(){ printf '\n===== %s  %s =====\n' "$*" "$(date '+%F %T')" | tee -a "$LOG"; }

say "QUEUE START"

# ======================================================================== A
say "A: feature labelling (top-activating residues)"
mkdir -p results_feature_labels
for root in outputs_ctrl outputs_ctrl_shuf outputs_ctrl_500tpp; do
  [ -d "$root" ] || continue
  for arm in $(ls "$root" 2>/dev/null | grep '^ckpt'); do
    for L in 11 14 18; do
      LD="$root/$arm/layer_$L"
      [ -f "$LD/Z.npy" ] || continue
      out="results_feature_labels/${root#outputs_}_${arm}_L${L}.csv"
      [ -f "$out" ] && { echo "  [$root $arm L$L] cached" | tee -a "$LOG"; continue; }
      echo "--- $root $arm L$L $(date '+%T')" | tee -a "$LOG"
      "$PY" label_features.py --layer-dir "$LD" --out "$out" >> "$LOG" 2>&1 \
        || echo "  !! failed $root $arm L$L" | tee -a "$LOG"
    done
  done
done
say "A done: $(ls results_feature_labels/*.csv 2>/dev/null | wc -l) cells"

# ======================================================================== B
say "B: shuffled corpus, seeds 43 and 44 (660e6 tokens, warmup 500)"
if [ ! -f "$SHUF/tokens.npy" ]; then
  echo "  !! no shuffled corpus at $SHUF -- skipping B and C" | tee -a "$LOG"
else
  for s in 43 44; do
    for obj in clm mlm; do
      case "$obj" in clm) name="ckpt_clm_s${s}";; mlm) name="ckpt_mlm_s${s}_token";; esac
      D="$SHUF/$name"
      [ -f "$D/model_final.pt" ] && { echo "  [$name] done - skip" | tee -a "$LOG"; continue; }
      mkdir -p "$D"
      res=""; [ -f "$D/model_resume.pt" ] && res="--resume $D/model_resume.pt"
      echo "--- train $name $(date '+%T')" | tee -a "$LOG"
      "$PY" -u train_ctrl_plm.py --objective "$obj" --data-dir "$SHUF" --out-dir "$D" \
        --seed "$s" --target-tokens "$TARGET" --batch-size "$BATCH" --seq-len "$SEQ" \
        --lr "$LR" --warmup "$WARMUP" --ckpt-every "$CKPT_EVERY" --val-every "$VAL_EVERY" \
        --d-model "$D_MODEL" --n-heads "$N_HEADS" --n-layers "$N_LAYERS" $res \
        >> "$LOG" 2>&1 || echo "  !! train failed $name" | tee -a "$LOG"
    done
  done
fi
say "B done"

# ======================================================================== C
say "C: shuffled corpus pair at 500 tok/param (21e9 tokens)"
if [ ! -f "$SHUF/tokens.npy" ]; then
  echo "  !! no shuffled corpus -- skipping C" | tee -a "$LOG"
else
  mkdir -p "$ABL_SHUF"
  for obj in clm mlm; do
    case "$obj" in clm) name="ckpt_clm_s42";; mlm) name="ckpt_mlm_s42_token";; esac
    D="$ABL_SHUF/$name"
    [ -f "$D/model_final.pt" ] && { echo "  [$name] done - skip" | tee -a "$LOG"; continue; }
    mkdir -p "$D"
    res=""; [ -f "$D/model_resume.pt" ] && res="--resume $D/model_resume.pt"
    echo "--- train 500tpp $name $(date '+%T')" | tee -a "$LOG"
    "$PY" -u train_ctrl_plm.py --objective "$obj" --data-dir "$SHUF" --out-dir "$D" \
      --seed 42 --target-tokens 21e9 --batch-size "$BATCH" --seq-len "$SEQ" \
      --lr "$LR" --warmup "$WARMUP" --ckpt-every 5000 --val-every 2000 --rolling-resume \
      --d-model "$D_MODEL" --n-heads "$N_HEADS" --n-layers "$N_LAYERS" $res \
      >> "$LOG" 2>&1 || echo "  !! train failed 500tpp $name" | tee -a "$LOG"
  done
fi
say "C done"

say "QUEUE COMPLETE"
echo "  feature-label cells : $(ls results_feature_labels/*.csv 2>/dev/null | wc -l)" | tee -a "$LOG"
echo "  shuffled 660e6      : $(ls -d $SHUF/ckpt_*_s4[34]* 2>/dev/null | wc -l)/4" | tee -a "$LOG"
echo "  shuffled 500tpp     : $(find $ABL_SHUF -name model_final.pt 2>/dev/null | wc -l)/2" | tee -a "$LOG"
