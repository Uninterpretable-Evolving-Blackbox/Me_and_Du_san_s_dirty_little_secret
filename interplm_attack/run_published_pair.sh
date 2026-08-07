#!/usr/bin/env bash
# run_published_pair.sh — InterPLM's own concept-F1 on the published ESM-2 / RITA
# pair, their code unmodified, only the backbone swapped.
#
# WHY THIS PAIR
#   Every published pLM SAE paper evaluates on ESM-2, which is masked. Not one
#   uses a causal pLM. RITA_l is causal, published, and a near-exact size match
#   for ESM-2-650M (680M vs 650M, 24 vs 33 blocks). It is the closest thing to a
#   controlled objective comparison that exists among released models -- and
#   unlike our 42M pair, anyone can download both halves and check.
#
# LAYERS
#   ESM-2-650M layer 18 is InterPLM's own choice: their released dictionary
#   (Elana/InterPLM-esm2-650m) is trained there. 18/33 = 54.5% relative depth;
#   0.545 * 24 = 13.1, so RITA_l layer 13 is the matched depth.
#
# SAE RECIPE
#   Expansion 8 and L1 0.06. Expansion 8 matches the dictionary InterPLM
#   released for ESM-2-650M (10,240 features on d_model 1280). Note this differs
#   from the expansion 4 used in RUN_INTERPLM_STRESS.sh, which follows their
#   *walkthrough* recipe for a 320-dim model -- different reference, deliberately.
#
# DISK
#   Embeddings are the constraint, not compute: 11.84M analysis residues at
#   d_model 1536 x float32 is 72 GB for RITA alone. Both arms held at once would
#   be ~205 GB against ~330 GB free. So each stage deletes its embeddings once
#   the artefact that depends on them exists. Peak is ~73 GB.
#
# SEEDS
#   One SAE per arm. The regenerated trainer routes the seed through
#   DataloaderConfig, but a single seed per arm means this run cannot speak to
#   dictionary-seed variance and must not be described as if it could.
set -u

BASE=${BASE:-$HOME/own_sae_data/interplm_stress}
REPO=$BASE/interplm_repo
PY=${PY:-$BASE/venv/bin/python}
HERE="$(cd "$(dirname "$0")" && pwd)"
ANN=$BASE/ann50k/processed
RES=$BASE/results_published_pair
DEV=${DEV:-mps}
BS=${BS:-4}
EXP=${EXP:-8}
L1=${L1:-0.06}
TRAIN_SHARDS=${TRAIN_SHARDS:-2}

mkdir -p "$RES"
[ -x "$PY" ] || { echo "!! no interpreter at $PY"; exit 2; }
[ -d "$ANN/valid" ] || { echo "!! $ANN/valid missing — prepare_eval_set never completed"; exit 2; }

# name : layer : d_model
SPECS=${SPECS:-"rita_l:13:1536 esm2_650m:18:1280"}

for spec in $SPECS; do
  NAME=${spec%%:*}; rest=${spec#*:}; LAYER=${rest%%:*}; DIM=${rest##*:}
  TAG=${NAME}_L${LAYER}
  T=$BASE/embd_train/$TAG
  A=$BASE/embd_analysis/$TAG
  SAVE=models/published/$TAG

  echo "########## $TAG  (d_model=$DIM, expansion $EXP -> $((DIM*EXP)) features)  $(date '+%F %T') ##########"

  # ---- 1. training embeddings -------------------------------------------
  if [ -f "$REPO/$SAVE/ae.pt" ]; then
    echo "  [1/6] SAE already exists — skipping embed+train"
  else
    echo "  [1/6] training embeddings ($TRAIN_SHARDS shards)"
    "$PY" "$HERE/embed_hf_interplm.py" --model "$NAME" --layer "$LAYER" \
        --fasta-dir "$BASE/train_shards" --limit-shards "$TRAIN_SHARDS" \
        --out "$T" --device "$DEV" --batch-size "$BS" --model-tag "$TAG" \
        2>&1 | tee -a "$RES/${TAG}_embed_train.log" | grep -vE "Loading weights|torch_dtype"
    [ -d "$T" ] || { echo "  !! no training embeddings"; exit 1; }

    # ---- 2. SAE, their trainer -------------------------------------------
    echo "  [2/6] SAE (their SAETrainingRun, unmodified library)"
    ( cd "$REPO" && CTRL_DIM=$DIM CTRL_EXP=$EXP CTRL_L1=$L1 CTRL_SEED=0 \
        CTRL_EMBD_DIR="$T" CTRL_SAVE_DIR="$SAVE" LAYER=$LAYER \
        "$PY" examples/train_ctrl_sae.py ) > "$RES/${TAG}_train_sae.log" 2>&1
    if [ ! -f "$REPO/$SAVE/ae.pt" ]; then
      echo "  !! SAE training failed — tail of log:"; tail -20 "$RES/${TAG}_train_sae.log"; exit 1
    fi
    rm -rf "$T"      # ~16 GB back before the analysis embeddings land
  fi

  # ---- 3. analysis embeddings -------------------------------------------
  if [ -f "$A/.done" ]; then
    echo "  [3/6] analysis embeddings already complete"
  else
    echo "  [3/6] analysis embeddings (8 shards, ~11.8M residues)"
    "$PY" "$HERE/embed_hf_interplm.py" --model "$NAME" --layer "$LAYER" \
        --shards-dir "$ANN" --out "$A" --device "$DEV" --batch-size "$BS" \
        --model-tag "$TAG" 2>&1 | tee -a "$RES/${TAG}_embed_analysis.log" \
        | grep -vE "Loading weights|torch_dtype"
    [ -f "$A/.done" ] || { echo "  !! analysis embeddings incomplete"; exit 1; }
  fi

  # ---- 4. outcome-blind quality FIRST, before any concept number ---------
  echo "  [4/6] dictionary quality (outcome-blind, before any F1 exists)"
  "$PY" "$HERE/sae_quality.py" --sae "$REPO/$SAVE/ae.pt" --embd "$A" \
      --repo "$REPO" --tag "$TAG" --header >> "$RES/sae_quality.txt" 2>&1
  tail -2 "$RES/sae_quality.txt"

  # ---- 5. their metric, unmodified --------------------------------------
  echo "  [5/6] normalize + compare_activations + calculate_f1"
  ( cd "$REPO" && "$PY" -m interplm.sae.normalize --sae_dir "$SAVE" \
      --aa_embds_dir "$A" ) > "$RES/${TAG}_normalize.log" 2>&1
  for SP in valid test; do
    ( cd "$REPO" && "$PY" -m interplm.analysis.concepts.compare_activations \
        --sae_dir "$SAVE" --aa_embds_dir "$A" \
        --eval_set_dir "$ANN/$SP/" --output_dir "results/published/${TAG}_${SP}" \
      && "$PY" -m interplm.analysis.concepts.calculate_f1 \
        --eval_res_dir "results/published/${TAG}_${SP}" \
        --eval_set_dir "$ANN/$SP/" ) > "$RES/${TAG}_score_${SP}.log" 2>&1 \
      || { echo "  !! scoring failed on $SP — tail:"; tail -20 "$RES/${TAG}_score_${SP}.log"; }
  done

  # ---- 6. report ---------------------------------------------------------
  echo "  [6/6] report_metrics"
  { echo "----- $TAG -----"
    ( cd "$REPO" && "$PY" -m interplm.analysis.concepts.report_metrics \
        --valid_path "results/published/${TAG}_valid/concept_f1_scores.csv" \
        --test_path  "results/published/${TAG}_test/concept_f1_scores.csv" ) 2>&1
  } | tee -a "$RES/concept_f1.txt"

  rm -rf "$A"        # free ~70 GB before the next arm
  echo "  done $TAG  $(date '+%F %T')"
done

echo "########## COMPLETE -> $RES ##########"
cat "$RES/concept_f1.txt"
