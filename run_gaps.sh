#!/usr/bin/env bash
# run_gaps.sh -- three rescoring stages over checkpoints that already exist.
#
# Single GPU, no pretraining, no new checkpoints. Everything here reuses models
# that already exist on this box. Safe to interrupt and re-run: every stage skips
# cells whose output file is already present.
#
#   STAGE=1  permuted-input rescore     18 cells   ~40 min
#   STAGE=2  random-init, seeds 43/44   12 cells   ~30 min
#   STAGE=3  fold-disjoint rescore      18 cells   ~40 min
#
#   bash run_gaps.sh              # all three, in order
#   STAGE=1 bash run_gaps.sh      # just one
#
# Each stage prints what to send back at the end.
set -uo pipefail

PY="${PY:-python}"
SEEDS="${SEEDS:-42 43 44}"
DEPTHS="${DEPTHS:-11 14 18}"
K_SPARSE="${K_SPARSE:-256}"
EXPANSION="${EXPANSION:-8}"
STAGE="${STAGE:-all}"

# Checkpoint roots. NATIVE_ROOT holds models trained on real sequences,
# SHUF_ROOT models trained on the order-destroyed corpus.
NATIVE_ROOT="${NATIVE_ROOT:-$HOME/own_sae_data/uniref50_pilot}"
SHUF_ROOT="${SHUF_ROOT:-$HOME/own_sae_data/uniref50_pilot_shuf}"

arms_for () { for s in $SEEDS; do echo "ckpt_mlm_s${s}_token"; echo "ckpt_clm_s${s}"; done; }

banner () { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }

# ---------------------------------------------------------------- preflight
[ -d "$NATIVE_ROOT" ] || echo "preflight: no $NATIVE_ROOT -- set NATIVE_ROOT"
[ -d "$SHUF_ROOT" ]   || echo "preflight: no $SHUF_ROOT -- set SHUF_ROOT (stages 1 and 3 need it)"
[ -d eval_set ]       || { echo "preflight: no eval_set/ -- cannot run"; exit 1; }

# ============================================================ STAGE 1
# Rescore the shuffled-corpus checkpoints on an order-permuted evaluation set.
# Builds eval_set_evalshuf/ if absent. Scoring code, structures and validation
# split are unchanged; only the evaluation input differs from the existing run.
stage_evaldist () {
  local rc=0 n=0
  [ -d eval_set_evalshuf ] || $PY -u make_eval_shuffled.py --eval-set eval_set --out eval_set_evalshuf --seed 42 || return 1
  for arm in $(arms_for); do
    local CK="$SHUF_ROOT/$arm/model_final.pt"
    [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
    for L in $DEPTHS; do
      local DEST="outputs_ctrl_evaldist/$arm/layer_$L"
      [ -f "$DEST/struct_seq_metrics.csv" ] && { echo "  [done] $arm L$L"; n=$((n+1)); continue; }
      echo "=== [evaldist] $arm L$L $(date +%H:%M:%S) ==="
      $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$arm" --layer "$L" \
          --out-root outputs_ctrl_evaldist --eval-set eval_set_evalshuf \
          --k-sparse "$K_SPARSE" --expansion "$EXPANSION" || { rc=1; continue; }
      $PY -u cpu_stage.py --layer-dir "$DEST" --model-type residue --n-shuffles 5 || rc=1
      n=$((n+1))
    done
  done

  if [ "$n" -eq 0 ]; then
    echo "  NO cells produced. A checkpoint root is wrong, or the eval set is missing."
    echo "  Not a silent skip: please say so rather than reporting this stage as done."
    rc=1
  fi
  echo; echo "  send back: outputs_ctrl_evaldist/*/*/struct_seq_metrics.csv"
  echo "             eval_set_evalshuf/META.json"
  return $rc
}

# ============================================================ STAGE 2
# Random-initialisation condition at two further seeds (43, 44), both arms.
# --randomize-model discards the trained weights and seeds off --sae-seed, so
# the checkpoint is read for its architecture only. No pretraining.
stage_untrained () {
  local rc=0 n=0
  for s in 43 44; do
    for arm in "ckpt_mlm_s42_token" "ckpt_clm_s42"; do
      local CK="$NATIVE_ROOT/$arm/model_final.pt"      # architecture only; weights discarded
      [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
      for L in $DEPTHS; do
        local DEST="outputs_ctrl_randominit_s${s}/$arm/layer_$L"
        [ -f "$DEST/struct_seq_metrics.csv" ] && { echo "  [done] s$s $arm L$L"; n=$((n+1)); continue; }
        echo "=== [untrained s$s] $arm L$L $(date +%H:%M:%S) ==="
        $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$arm" --layer "$L" \
            --out-root "outputs_ctrl_randominit_s${s}" --eval-set eval_set \
            --randomize-model --sae-seed "$s" \
            --k-sparse "$K_SPARSE" --expansion "$EXPANSION" || { rc=1; continue; }
        $PY -u cpu_stage.py --layer-dir "$DEST" --model-type residue --n-shuffles 5 || rc=1
      n=$((n+1))
      done
    done
  done

  if [ "$n" -eq 0 ]; then
    echo "  NO cells produced. A checkpoint root is wrong, or the eval set is missing."
    echo "  Not a silent skip: please say so rather than reporting this stage as done."
    rc=1
  fi
  echo; echo "  send back: outputs_ctrl_randominit_s43/*/*/struct_seq_metrics.csv"
  echo "             outputs_ctrl_randominit_s44/*/*/struct_seq_metrics.csv"
  return $rc
}

# ============================================================ STAGE 3
# Extend the fold-disjoint evaluation set to the shuffled-corpus checkpoints.
# run_checks.sh stage 3 has its checkpoint root hardcoded and covers only the
# native arm, so this runs the other half. Needs eval_set_folddisj/.
stage_folddisj_shuf () {
  local rc=0 n=0
  [ -d eval_set_folddisj ] || { echo "  no eval_set_folddisj/ -- run: FOLDDISJ_APPLY=1 ONLY=3 bash run_checks.sh"; return 1; }
  for arm in $(arms_for); do
    local CK="$SHUF_ROOT/$arm/model_final.pt"
    [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
    for L in $DEPTHS; do
      local DEST="outputs_ctrl_folddisj_shuf/$arm/layer_$L"
      [ -f "$DEST/struct_seq_metrics.csv" ] && { echo "  [done] $arm L$L"; n=$((n+1)); continue; }
      echo "=== [folddisj-shuf] $arm L$L $(date +%H:%M:%S) ==="
      $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$arm" --layer "$L" \
          --out-root outputs_ctrl_folddisj_shuf --eval-set eval_set_folddisj \
          --k-sparse "$K_SPARSE" --expansion "$EXPANSION" || { rc=1; continue; }
      $PY -u cpu_stage.py --layer-dir "$DEST" --model-type residue --n-shuffles 5 || rc=1
      n=$((n+1))
    done
  done

  if [ "$n" -eq 0 ]; then
    echo "  NO cells produced. A checkpoint root is wrong, or the eval set is missing."
    echo "  Not a silent skip: please say so rather than reporting this stage as done."
    rc=1
  fi
  echo; echo "  send back: outputs_ctrl_folddisj_shuf/*/*/struct_seq_metrics.csv"
  return $rc
}

# ---------------------------------------------------------------- dispatch
FAILED=0
run_one () {
  banner "$2"
  if $1; then echo "  [$2] ok"; else local rc=$?; echo "  [$2] FAILED (rc=$rc)"; FAILED=1; fi
}

case "$STAGE" in
  1)   run_one stage_evaldist        "STAGE 1  permuted-input rescore" ;;
  2)   run_one stage_untrained       "STAGE 2  random-init, seeds 43/44" ;;
  3)   run_one stage_folddisj_shuf   "STAGE 3  fold-disjoint rescore" ;;
  all) run_one stage_evaldist        "STAGE 1  permuted-input rescore"
       run_one stage_untrained       "STAGE 2  random-init, seeds 43/44"
       run_one stage_folddisj_shuf   "STAGE 3  fold-disjoint rescore" ;;
  *)   echo "unknown STAGE=$STAGE (use 1, 2, 3 or all)"; exit 2 ;;
esac

banner "done"
echo "Please also paste, from this directory:"
echo "    git rev-parse HEAD"
echo "    git status --short"
echo "so the numbers can be tied to the code that produced them."
exit $FAILED
