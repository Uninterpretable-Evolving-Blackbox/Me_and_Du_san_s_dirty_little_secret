#!/usr/bin/env bash
# run_crosscoder_ctrl.sh — the MLM-vs-CLM model-diffing crosscoder. THE confirmatory experiment.
#
# WHAT IT ANSWERS
#   Everything so far trained a SEPARATE feature dictionary per arm and compared statistics of
#   the two sets. The dictionaries have no correspondence — feature #57 in one is unrelated to
#   #57 in the other — so the question we actually care about was never expressible:
#
#       Are the features encoding 3D structure SHARED between the two training objectives,
#       or do they exist ONLY in the masked model?
#
#   A crosscoder trains ONE dictionary that must reconstruct BOTH arms, with a separate output
#   matrix per arm. Every feature is now a single shared object, and how much each arm relies on
#   it is a per-feature measurement rather than a distribution comparison.
#
# WHY THIS PAIR IS THE PRIMARY RESULT (not the older ESM-2-vs-RITA one)
#   ESM-2 vs RITA differs in architecture, objective, data AND scale at once, so it cannot
#   attribute anything to the objective — it is the exploratory pair. Here only the objective
#   differs: one backbone, one initialisation, one data order. This is where the causal claim
#   is made, so the hyperparameters are FROZEN in advance (tuned on the exploratory pair) and
#   nothing here is tuned on the outcome.
#
#   It is also the only pair where the field's validity diagnostic works. Latent Scaling
#   (Minder et al. 2025, Eq. 4) dots the assigned arm's decoder into the OTHER arm's activation
#   space, so it needs equal widths: fine at d=320/d=320, undefined at 1280 vs 1536.
#
# STAGES — please run stage 1, send results, THEN stage 2.
#   STAGE=0  ~15 min   one cell. Sanity + the go/no-go check (see README).
#   STAGE=1  ~1-2 h    seed 42 across depths.            <- default
#   STAGE=2  ~2-4 h    seeds 43 and 44. ONLY after we have looked at stage 1.
#
#   bash run_crosscoder_ctrl.sh                      # stage 1 (seed 42)
#   STAGE=0 bash run_crosscoder_ctrl.sh              # the 15-minute check first
#   STAGE=2 bash run_crosscoder_ctrl.sh              # the other two seeds, once Wei says go
#
# HYPERPARAMETERS ARE FROZEN. Do not tune them here — they were selected on the exploratory
# pair using reconstruction quality only (never on the result). Wei will send the three numbers;
# until then the defaults below are the project's SAE convention.
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
OUT="${OUT:-outputs_crosscoder}"
ARM_A="${ARM_A:-mlm_token}"
ARM_B="${ARM_B:-clm}"
STAGE="${STAGE:-1}"

# --- frozen from the exploratory-pair ablation (results_crosscoder_ablation/selected.json) ---
EXPANSION="${EXPANSION:-8}"
K_FRAC="${K_FRAC:-0.20}"
EPOCHS="${EPOCHS:-60}"
# ---------------------------------------------------------------------------------------------

# C1 (the validity check) found the MLM SAEs degenerate at blocks 0 and 4, so those depths are
# not trustworthy for either arm's comparison. 7/14/22 = 25%/50%/75% relative depth.
DEPTHS_MAIN="${DEPTHS:-7 14 22}"
DEPTHS_FULL="0 4 7 11 14 18 22 26 29"

ckpt_dir () {
  case "$1" in clm) echo "ckpt_clm_s42";; mlm_token) echo "ckpt_mlm_s42_token";;
               mlm_pred) echo "ckpt_mlm_s42_pred";; *) echo "ckpt_$1";; esac
}
CKA="$DATA/$(ckpt_dir "$ARM_A")/model_final.pt"
CKB="$DATA/$(ckpt_dir "$ARM_B")/model_final.pt"
[ -f "$CKA" ] || { echo "ERROR: no checkpoint at $CKA"; exit 1; }
[ -f "$CKB" ] || { echo "ERROR: no checkpoint at $CKB"; exit 1; }
[ -d cache/pdb_files ] || { echo "ERROR: no cache/pdb_files (run fetch_pdbs.py). L_struct needs it."; exit 1; }
$PY - <<'PY' || exit 1
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow)")
PY

cell () {  # layer, seed
  local L="$1"
  local S="$2"
  local D="$OUT/cc_${ARM_A}_${ARM_B}_l${L}_seed${S}"
  if [ -f "$D/struct_seq_metrics.csv" ] && grep -q spearman_dn_lstruct "$D/META.json" 2>/dev/null; then
    echo "  [done] L$L seed $S"; return 0
  fi
  echo "=== crosscoder ${ARM_A} vs ${ARM_B} | L$L | seed $S | $(date +%H:%M:%S) ==="
  if [ ! -f "$D/latent_diffing.csv" ]; then
    $PY -u eval_crosscoder_ctrl.py --ckpt-a "$CKA" --ckpt-b "$CKB" \
        --name-a "$ARM_A" --name-b "$ARM_B" --layer "$L" --seed "$S" \
        --expansion "$EXPANSION" --k-frac "$K_FRAC" --epochs "$EPOCHS" \
        --out-root "$OUT" --eval-set eval_set || { echo "!! eval L$L s$S"; return 1; }
  fi
  # --n-shuffles 5 is REQUIRED: cpu_stage defaults to 3, but L_struct is shuffle-corrected and
  # the paper uses 5 — 3 would compute a different metric than every other number in the project.
  if [ ! -f "$D/struct_seq_metrics.csv" ]; then
    $PY -u cpu_stage.py --layer-dir "$D" --model-type residue --n-shuffles 5 \
        --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
        --fasta-path cache/scope_40.fa || { echo "!! cpu_stage L$L s$S"; return 1; }
  fi
  $PY -u eval_crosscoder_ctrl.py --crosstab "$D" || echo "!! crosstab L$L s$S"
  [ "${KEEP_Z:-0}" = "1" ] || rm -f "$D/Z.npy"   # L_struct + crosstab done; Z not needed further
}

echo "########## CROSSCODER ${ARM_A} vs ${ARM_B} | STAGE $STAGE | $(date) ##########"
echo "  frozen hyperparameters: expansion=$EXPANSION k_frac=$K_FRAC epochs=$EPOCHS"

case "$STAGE" in
  0) echo "  stage 0: single sanity/go-no-go cell (L14, seed 42)"
     cell 14 42 ;;
  1) echo "  stage 1: seed 42 across depths $DEPTHS_MAIN"
     for L in $DEPTHS_MAIN; do cell "$L" 42; done
     if [ "${FULL_PROFILE:-0}" = "1" ]; then
       echo "  + full 9-depth profile at seed 42"
       for L in $DEPTHS_FULL; do cell "$L" 42; done
     fi ;;
  2) echo "  stage 2: seeds 43 and 44 across depths $DEPTHS_MAIN"
     for S in 43 44; do for L in $DEPTHS_MAIN; do cell "$L" "$S"; done; done ;;
  *) echo "unknown STAGE=$STAGE (use 0, 1 or 2)"; exit 1 ;;
esac

echo; echo "########## STAGE $STAGE DONE $(date) ##########"
if [ "$STAGE" = "0" ]; then
  echo "Please send Wei these two lines from above, then STOP:"
  echo "   Delta_norm median ... | absolute bins occupied n/5"
  echo "   Spearman(Delta_norm, struct_delta) = ...   <-- PRIMARY"
else
  echo "Send back (small — no Z):"
  find "$OUT" -name 'latent_diffing.csv' -o -name 'META.json' -o -name 'struct_seq_metrics.csv' \
    2>/dev/null | sort | sed 's/^/  /'
  echo
  echo "  tar czf crosscoder_stage${STAGE}.tgz \$(find $OUT -name 'latent_diffing.csv' -o -name 'META.json' -o -name 'struct_seq_metrics.csv')"
  [ "$STAGE" = "1" ] && echo && echo "STOP HERE. Wei looks at stage 1 before you run STAGE=2 (the other two seeds)."
fi
