#!/usr/bin/env bash
# run_ctrl_rigor.sh — the RTX half of the rigor program, in priority order.
#
# Each stage is separate and gated: run it, send results, wait. Nothing here trains a new model
# except STAGE=3, which is the single most valuable run in the whole project.
#
#   STAGE=1  C7  frozen-SAE null baselines      ~3-5 h   analysis only
#   STAGE=2  A8  k x expansion sweep            ~2-4 h   analysis only
#   STAGE=3  C5  TRAINING SEEDS 43 and 44       ~30-40 h TRAINS MODELS — the priority
#   STAGE=4  E1  crosscoder, seed 42            ~1-2 h   analysis only
#   STAGE=5  E2  crosscoder, seeds 43/44        ~2-4 h   analysis only
#
#   bash run_ctrl_rigor.sh              # stage 1
#   STAGE=3 bash run_ctrl_rigor.sh      # the multi-seed run
#
# WHY STAGE 3 MATTERS MOST. Right now there is exactly ONE masked model and ONE causal model.
# Every difference we report — locality, effective rank, concept alignment — compares two
# individual training runs, with no way to tell a real effect from run-to-run variation. Protein
# bootstraps do not fix this; they resample proteins, not training runs. Two more seeds per arm
# turns "two models" into an experiment. If only one thing on this list gets run, make it this.
set -u
cd "$(dirname "$0")"

# Stamp the running code's version into the log. `git pull` is a purely local operation, so
# there is no way to tell from GitHub which revision a collaborator is actually running — this
# makes every log self-identifying, so "did you pull?" is answered by any output they send back.
_REV=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo " +local-changes")
echo "########## repo revision: ${_REV}${_DIRTY} | $(date) ##########"
if [ "$_REV" != "unknown" ] && ! git merge-base --is-ancestor 60fca90 HEAD 2>/dev/null; then
  echo "!! WARNING: this revision PREDATES commit 60fca90, which fixes a cleanup bug that can"
  echo "!! delete results it never verified. Run 'git pull' before this finishes. Pulling"
  echo "!! mid-run is safe — the scripts skip work that is already done."
fi
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
STAGE="${STAGE:-1}"
ARM_A="${ARM_A:-mlm_token}"
ARM_B="${ARM_B:-clm}"
DEPTHS="${DEPTHS:-7 14 22}"        # 25/50/75% — C1 says blocks 0 and 4 are untrustworthy
SWEEP_LAYER="${SWEEP_LAYER:-14}"

ckpt_dir () {
  case "$1" in clm) echo "ckpt_clm_s42";; mlm_token) echo "ckpt_mlm_s42_token";;
               mlm_pred) echo "ckpt_mlm_s42_pred";; *) echo "ckpt_$1";; esac
}
[ -d cache/pdb_files ] || { echo "ERROR: no cache/pdb_files (run fetch_pdbs.py)."; exit 1; }
$PY - <<'PY' || exit 1
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow)")
PY

# ---------------- STAGE 1: C7 frozen-component null baselines ----------------
stage_c7 () {
  echo "########## C7 frozen-SAE null baselines ##########"
  echo "  Does a RANDOM dictionary reproduce the MLM-vs-CLM difference? If it does, the SAE's"
  echo "  learning is not what produces the result. (Korznikov et al. 2026 showed frozen SAEs"
  echo "  match trained ones on interpretability, probing and causal editing.)"
  for arm in "$ARM_A" "$ARM_B"; do
    local CK="$DATA/$(ckpt_dir "$arm")/model_final.pt"
    [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
    for fz in frozen_decoder soft_frozen_decoder frozen_encoder; do
      for L in $DEPTHS; do
        local NAME="ctrl_${arm}_ptn_${fz}"
        local LD="outputs_ctrl/$NAME/layer_$L"
        [ -f "$LD/struct_seq_metrics.csv" ] && { echo "  [done] $NAME L$L"; continue; }
        echo "=== $NAME L$L $(date +%H:%M:%S) ==="
        $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$NAME" --layer "$L" \
            --out-root outputs_ctrl --eval-set eval_set --per-token-norm --freeze "$fz" \
          || { echo "!! eval $NAME L$L"; continue; }
        $PY -u cpu_stage.py --layer-dir "$LD" --model-type residue --n-shuffles 5 \
            --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
            --fasta-path cache/scope_40.fa || echo "!! cpu_stage $NAME L$L"
        [ "${KEEP_Z:-0}" = "1" ] || rm -f "$LD/Z.npy"
      done
    done
  done
}

# ---------------- STAGE 2: A8 k x expansion sweep ----------------
stage_a8 () {
  echo "########## A8 k x expansion sweep on the CONTROLLED models ##########"
  echo "  The existing 72-run sweep is all ESM-2/RITA. Transferring it here has failed twice"
  echo "  (k/embed 53%, then k=96). This asks whether ANY config puts both arms in the same"
  echo "  regime — if none does, that is the rank gap and it is a finding, not a tuning problem."
  $PY -u sweep_ctrl_sae.py --layer "$SWEEP_LAYER" --arms "$ARM_A,$ARM_B" \
      --ks 64,128,256 --expansions 4,8,16,32 || echo "!! sweep failed"
}

# ---------------- STAGE 3: C5 training seeds (the priority) ----------------
stage_c5 () {
  echo "########## C5 TRAINING SEEDS 43 and 44 — trains models, ~30-40 h ##########"
  echo "  Turns n=1 per arm into n=3. Resumable: interrupt freely, re-run to continue."
  SEEDS="43 44" bash run_full_ctrl.sh
}

# ---------------- STAGES 4/5: crosscoder ----------------
stage_cc () {
  local ST="$1"
  echo "########## crosscoder stage $ST ##########"
  STAGE="$ST" bash run_crosscoder_ctrl.sh
}

case "$STAGE" in
  1) stage_c7 ;;
  2) stage_a8 ;;
  3) stage_c5 ;;
  4) stage_cc 1 ;;
  5) stage_cc 2 ;;
  *) echo "unknown STAGE=$STAGE (1..5)"; exit 1 ;;
esac

echo; echo "########## STAGE $STAGE DONE $(date) ##########"
echo "Send back (small, no Z):"
find outputs_ctrl -name struct_seq_metrics.csv -o -name META.json 2>/dev/null | grep ctrl_ | sort | sed 's/^/  /' | head -40
[ -f results_ctrl_sweep/sweep.csv ] && echo "  results_ctrl_sweep/sweep.csv"
find outputs_robustness -name 'bootstrap_h1_ctrl_esmc_s4[34]_*.csv' 2>/dev/null | sort | sed 's/^/  /'
