#!/usr/bin/env bash
# run_all_pending.sh — EVERYTHING outstanding, in dependency order, one command.
#
#   bash run_all_pending.sh
#
# ~24-27 h total. Every stage is idempotent and skip-if-done, so interrupting it costs at
# most the cell in flight — re-run the same command and it picks up where it stopped.
#
# A FAILING STAGE NEVER STOPS THE QUEUE. Later stages do not depend on earlier ones
# (the one real dependency, stage 7's gate, is checked inside stage 7), so a crash at 3am
# must not idle the GPU until morning. Failures are collected and printed at the end.
#
#   1  mechanism: upstream vs downstream contacts     ~2 h    the only mechanism test we have
#   2  frozen-dictionary nulls, controlled pair       ~3-5 h  the control that reframed the paper
#   3  prediction-matched protocol, seeds 43/44       ~12 h   closes the last single-seed claim
#      (--match-predictions divides steps by mask_rate: 660M/0.15 = 4.4B tokens per MLM
#       model, ~5.25 h each. Both seeds train from scratch; the CLM arms are shared
#       across protocols and skip. SKIP="3" if you would rather have the other six first.)
#   4  contact-definition sweep (cutoff x separation) ~1 h    closes the band question
#   5  SAE-free readout (probe + contacts)            ~2 h    no SAE anywhere; immune to stage 6
#   6  L_struct vs SAE capacity                       ~2.5 h  is the EV gap even relevant?
#   7  crosscoder: 15-min gate, then seed 42          ~1-2 h  gated; skips itself if the gate fails
#
# Run a single stage instead:  ONLY=3 bash run_all_pending.sh
# Skip stages:                 SKIP="7" bash run_all_pending.sh
set -u
cd "$(dirname "$0")"

_REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo " +local-changes")
echo "########## repo revision: ${_REV}${_DIRTY} | started $(date) ##########"

PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUT="${OUT:-outputs_ctrl}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
LOGDIR="logs_pending"; mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FAILED=""

# Capacity settings. The sweep showed no configuration equalises the two arms, so these
# stay at the values every existing controlled result used — changing them would invalidate
# the three C5 seeds for no gain. Stage 6 is what tests whether they matter at all.
K_SPARSE="${K_SPARSE:-256}"
EXPANSION="${EXPANSION:-8}"
DEPTHS_HEADLINE="${DEPTHS_HEADLINE:-11 14 18}"   # 38/50/63% — largest effect, neither arm degenerate
SEEDS="${SEEDS:-42 43 44}"

[ -d cache/pdb_files ] || { echo "ERROR: no cache/pdb_files (run fetch_pdbs.py)."; exit 1; }

want () {   # stage number -> should we run it?
  local n="$1"
  [ -n "$ONLY" ] && [ "$ONLY" != "$n" ] && return 1
  case " $SKIP " in *" $n "*) return 1 ;; esac
  return 0
}

run () {    # run <n> <name> <command...>
  local n="$1" name="$2"; shift 2
  want "$n" || { echo "  [skip stage $n] $name"; return 0; }
  local log="$LOGDIR/stage${n}_${STAMP}.log"
  echo
  echo "===== STAGE $n: $name — started $(date) -> $log ====="
  if "$@" > "$log" 2>&1; then
    echo "===== STAGE $n OK $(date) ====="
  else
    echo "===== STAGE $n FAILED (rc=$?) $(date) — CONTINUING ====="
    FAILED="$FAILED $n"
    tail -25 "$log" | sed 's/^/    /'
  fi
}

# ---------------------------------------------------------------- 1: mechanism
run 1 "mechanism — upstream vs downstream contacts" \
  env STAGE=1 K_SPARSE="$K_SPARSE" EXPANSION="$EXPANSION" bash run_ctrl_mechanism.sh

# ---------------------------------------------------------------- 2: frozen nulls
run 2 "frozen-dictionary nulls on the controlled pair" \
  env STAGE=1 bash run_ctrl_rigor.sh

# ---------------------------------------------------------------- 3: pred protocol
run 3 "prediction-matched protocol, seeds 43/44" \
  env PROTOCOLS=pred SEEDS="43 44" KEEP_Z=1 bash run_full_ctrl.sh

# ---------------------------------------------------------------- 4: contact definition
# Z.npy is pruned after the main controlled run, and this sweep recomputes struct_delta
# from Z. Stage 1 happens to regenerate Z at exactly these depths, so in the normal
# front-to-back order this is already satisfied — but ONLY=4 would then silently produce
# nothing. Regenerate here too if it is missing; it is a no-op when stage 1 already ran.
ensure_z () {   # ensure_z <arm> <layer>
  local arm="$1" L="$2"
  local LD="$OUT/$arm/layer_$L"
  [ -f "$LD/Z.npy" ] && return 0
  local CK="$DATA/$arm/model_final.pt"
  [ -f "$CK" ] || { echo "  [skip] no $CK"; return 1; }
  echo "=== [regen Z] $arm L$L $(date +%H:%M:%S) ==="
  $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$arm" --layer "$L" \
      --out-root "$OUT" --eval-set eval_set \
      --k-sparse "$K_SPARSE" --expansion "$EXPANSION"
}

stage_contactdef () {
  local rc=0
  for s in $SEEDS; do
    for arm in "ckpt_mlm_s${s}_token" "ckpt_clm_s${s}"; do
      for L in $DEPTHS_HEADLINE; do ensure_z "$arm" "$L" || rc=1; done
    done
  done
  for s in $SEEDS; do
    $PY -u experiment_contact_def_sweep.py --root "$OUT" \
        --model-a "ckpt_mlm_s${s}_token" --model-b "ckpt_clm_s${s}" \
        --layers "$(echo $DEPTHS_HEADLINE | tr ' ' ',')" \
        --out "results_contact_def_sweep_s${s}" || rc=1
  done
  return $rc
}
run 4 "contact-definition sweep (Ca 6/8/10 x |i-j| 6/12/24)" stage_contactdef

# ---------------------------------------------------------------- 5: SAE-free
stage_saefree () {
  local arms=""
  for s in $SEEDS; do
    arms="${arms:+$arms,}ckpt_mlm_s${s}_token,ckpt_clm_s${s}"
  done
  $PY -u eval_ctrl_saefree.py --ckpt-root "$DATA" --arms "$arms" \
      --depths 0,4,7,11,14,18,22,26,29 --out results_ctrl_saefree
}
run 5 "SAE-free readout on the controlled pair" stage_saefree

# ---------------------------------------------------------------- 6: capacity vs L_struct
# Six configs spanning the whole val_EV range the sweep found. The CLM arm is the point:
# capacity drives it from ~0.81 down to roughly zero, so if L_struct is flat across THAT,
# reconstruction quality is not what L_struct measures.
CAP_CONFIGS="${CAP_CONFIGS:-64:4 256:4 64:8 256:8 256:16 256:32}"
CAP_LAYER="${CAP_LAYER:-14}"
stage_capacity () {
  local rc=0
  for arm in "ckpt_mlm_s42_token" "ckpt_clm_s42"; do
    local CK="$DATA/$arm/model_final.pt"
    [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
    for cfg in $CAP_CONFIGS; do
      local K="${cfg%%:*}" E="${cfg##*:}"
      local NAME="${arm}_k${K}e${E}"
      local LD="outputs_ctrl_cap/$NAME/layer_$CAP_LAYER"
      [ -f "$LD/struct_seq_metrics.csv" ] && { echo "  [done] $NAME"; continue; }
      echo "=== $NAME (k=$K, expansion=$E) $(date +%H:%M:%S) ==="
      $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$NAME" --layer "$CAP_LAYER" \
          --out-root outputs_ctrl_cap --eval-set eval_set \
          --k-sparse "$K" --expansion "$E" || { rc=1; continue; }
      $PY -u cpu_stage.py --layer-dir "$LD" --model-type residue --n-shuffles 5 \
          --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
          --fasta-path cache/scope_40.fa || rc=1
      rm -f "$LD/Z.npy"     # this stage needs no bootstrap, so Z is pure disk cost
    done
  done
  $PY -u analyze_capacity_lstruct.py --root outputs_ctrl_cap --layer "$CAP_LAYER" || rc=1
  return $rc
}
run 6 "does L_struct depend on SAE capacity at all?" stage_capacity

# ---------------------------------------------------------------- 7: crosscoder (gated)
# The 15-minute gate exists because the Delta_norm dynamic range may be too small to
# support any claim. Run it, and only continue if it passes — never spend hours on spec.
stage_crosscoder () {
  echo "--- 15-minute go/no-go gate ---"
  if ! STAGE=0 bash run_crosscoder_ctrl.sh; then
    echo "GATE FAILED — not running the full crosscoder. This is the designed behaviour,"
    echo "not an error: the gate exists to stop us spending hours on an unusable measurement."
    return 0
  fi
  echo "--- gate passed, running seed 42 ---"
  STAGE=1 bash run_crosscoder_ctrl.sh
}
run 7 "crosscoder (gate, then seed 42 if the gate passes)" stage_crosscoder

# ---------------------------------------------------------------- report
echo
echo "########## ALL STAGES DONE $(date) ##########"
if [ -n "$FAILED" ]; then
  echo "FAILED STAGES:$FAILED   (logs in $LOGDIR/)"
else
  echo "no stage failures"
fi
echo
echo "Send back — everything below is small, no Z:"
{
  find "$OUT" -name struct_seq_metrics.csv -o -name META.json
  find outputs_ctrl_cap results_ctrl_saefree results_contact_def_sweep_s* \
       results_rigor results_concept_f1 -type f \( -name '*.csv' -o -name '*.json' \) 2>/dev/null
  find outputs_robustness -name '*.csv'
  ls "$LOGDIR"/*.log
} 2>/dev/null | sort | sed 's/^/  /' | head -60
echo
echo "  tar czf pending_results.tgz \\"
echo "    \$(find $OUT -name 'struct_seq_metrics.csv' -o -name 'META.json') \\"
echo "    outputs_ctrl_cap results_ctrl_saefree results_contact_def_sweep_s* \\"
echo "    results_rigor results_concept_f1 outputs_robustness/*.csv $LOGDIR"
