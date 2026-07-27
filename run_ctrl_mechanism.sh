#!/usr/bin/env bash
# run_ctrl_mechanism.sh — the two experiments that change what the paper can claim.
#
# Both are NEW code (2026-07-27). Neither touches an existing result: stage 1 writes a
# separate CSV per cell, stage 2 writes to a separate corpus and output root.
#
#   STAGE=1  MECHANISM   directional contact split          ~2 h   needs Z
#   STAGE=2  VALIDITY    residue-shuffled pretraining       ~3 h   trains 2 models
#
# WHY THESE TWO. Everything else on the list defends a result that already exists.
# These two decide what the result MEANS:
#   * Stage 1 asks whether the masked model's advantage sits on contacts a causal model
#     structurally cannot see. If yes, the paper has a mechanism tied to the manipulated
#     variable instead of an association. If no, the bidirectional-access explanation is
#     refuted — also worth knowing, and worth reporting.
#   * Stage 2 asks whether L_struct is zero when it ought to be. The metric is homemade;
#     nothing currently establishes it responds to real structure rather than to
#     composition, length or position. Train on residue-shuffled sequences, evaluate on
#     REAL proteins and REAL structures, and L_struct should collapse.
#
# ORDERING. Run the k x expansion sweep (run_ctrl_rigor.sh STAGE=2) FIRST. If it changes
# k_sparse, set K_SPARSE below to the new value before running either stage here, or the
# numbers will not be comparable to the rest of the controlled results.
set -u
cd "$(dirname "$0")"

_REV=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo " +local-changes")
echo "########## repo revision: ${_REV}${_DIRTY} | $(date) ##########"

PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUT="${OUT:-outputs_ctrl}"
STAGE="${STAGE:-1}"
K_SPARSE="${K_SPARSE:-256}"          # <-- update from the sweep result before running
EXPANSION="${EXPANSION:-8}"
N_SHUFFLES=5                          # never 3: cpu_stage's default computes a DIFFERENT metric
# 38/50/63% — where the effect is largest and the SAE is not degenerate at either arm.
DEPTHS="${DEPTHS:-11 14 18}"
SEEDS="${SEEDS:-42 43 44}"
REF_LAYER_DIR="${REF_LAYER_DIR:-outputs_layerwise/esm2/layer_16}"

[ -d cache/pdb_files ] || { echo "ERROR: no cache/pdb_files (run fetch_pdbs.py)."; exit 1; }

# ---------------- STAGE 1: the mechanism test ----------------
stage_mechanism () {
  echo "########## MECHANISM: upstream vs downstream contacts ##########"
  echo "  A causal model cannot see j>i. If bidirectionality is the mechanism, the MLM"
  echo "  advantage must be LARGER on downstream contacts. The sequential graph is split"
  echo "  the same way as a within-run control for a generic forward-looking advantage."
  echo
  local MLM_CELLS="" CLM_CELLS=""
  for s in $SEEDS; do
    for arm in "ckpt_mlm_s${s}_token" "ckpt_clm_s${s}"; do
      local CK="$DATA/$arm/model_final.pt"
      [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
      for L in $DEPTHS; do
        local LD="$OUT/$arm/layer_$L"
        if [ -f "$LD/struct_seq_directional.csv" ]; then
          echo "  [done] $arm L$L"; continue
        fi
        # Z is pruned after the main run, but the SAE seed is fixed and training is
        # deterministic, so re-running the extractor reproduces the SAME dictionary
        # (verify: val_EV in the rewritten META.json must match the original).
        if [ ! -f "$LD/Z.npy" ]; then
          echo "=== [regen Z] $arm L$L $(date +%H:%M:%S) ==="
          $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$arm" --layer "$L" \
              --out-root "$OUT" --eval-set eval_set \
              --k-sparse "$K_SPARSE" --expansion "$EXPANSION" \
            || { echo "!! regen failed $arm L$L"; continue; }
        fi
        echo "=== [directional] $arm L$L $(date +%H:%M:%S) ==="
        $PY -u cpu_stage.py --layer-dir "$LD" --model-type residue \
            --n-shuffles "$N_SHUFFLES" --split-direction updown \
            --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
            --fasta-path cache/scope_40.fa || echo "!! cpu_stage failed $arm L$L"
      done
    done
    MLM_CELLS="${MLM_CELLS:+$MLM_CELLS,}ckpt_mlm_s${s}_token"
    CLM_CELLS="${CLM_CELLS:+$CLM_CELLS,}ckpt_clm_s${s}"
  done

  echo
  echo "########## MECHANISM ANALYSIS ##########"
  $PY -u analyze_directional.py --root "$OUT" \
      --mlm "$MLM_CELLS" --clm "$CLM_CELLS" \
      --depths "$(echo $DEPTHS | tr ' ' ',')" \
      --out results_rigor/directional_mechanism.csv || echo "!! analysis failed"
}

# ---------------- STAGE 2: construct validity ----------------
stage_shuffled () {
  echo "########## CONSTRUCT VALIDITY: residue-shuffled pretraining ##########"
  echo "  Length and composition preserved exactly; all sequence order destroyed."
  echo "  Evaluated on the REAL proteins and REAL structures."
  echo "  PREDICTION: L_struct collapses toward 0 and the MLM-CLM gap vanishes."
  echo "  IF IT DOES NOT: L_struct is tracking composition/length/position, not structure,"
  echo "  and that must be reported before anything else in the paper is believed."
  echo
  local SHUF="$HOME/own_sae_data/uniref50_pilot_shuf"
  if [ ! -f "$SHUF/tokens.npy" ]; then
    echo "=== [prep shuffled corpus] $(date) ==="
    $PY -u prep_controlled_corpus.py --shuffle-residues \
        --n-sequences 3000000 || { echo "!! corpus prep failed"; return 1; }
  else
    echo "  [done] shuffled corpus already at $SHUF"
  fi
  # One seed is enough: this is a floor check, not an effect estimate.
  # EVERY output namespace is redirected. With SEEDS="42" PROTOCOLS="token" the default
  # bootstrap stem and concept-F1 path are byte-identical to the real run's, so without
  # BOOT_STEM_PREFIX and CF_ROOT this control would silently write over the C5 results.
  DATA="$SHUF" OUT="outputs_ctrl_shuf" PROTOCOLS="token" SEEDS="42" KEEP_Z=1 \
    BOOT_STEM_PREFIX="bootstrap_h1_ctrl_shuf" CF_ROOT="results_concept_f1_shuf" \
    bash run_full_ctrl.sh || echo "!! shuffled run failed"
  echo
  echo "  Compare mean struct_delta in outputs_ctrl_shuf/ against outputs_ctrl/."
  echo "  Report the ratio, not just the sign."
}

case "$STAGE" in
  1) stage_mechanism ;;
  2) stage_shuffled ;;
  *) echo "unknown STAGE=$STAGE (1=mechanism, 2=shuffled-control)"; exit 1 ;;
esac
echo; echo "########## STAGE $STAGE DONE $(date) ##########"
