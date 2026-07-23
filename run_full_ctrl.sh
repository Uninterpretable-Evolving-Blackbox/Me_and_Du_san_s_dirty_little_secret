#!/usr/bin/env bash
# run_full_ctrl.sh — the WHOLE controlled MLM-vs-CLM experiment, end to end.
#
#   train  ->  extract + SAE (9 depths)  ->  L_struct  ->  per-feature CSVs
#
# Everything stays on this machine; only small CSVs need to come back. The big
# intermediates (Z.npy ~1.5 GB/layer) are written to $OUT and can be deleted after.
#
# WHAT IS VARIED (the experiment):
#   --objective  mlm | clm      (attention mask + loss)   <- the variable
#   --seed       42 | 43 | 44   (init)                    <- replication
#   PROTOCOL     token | pred   (fair-compute definition) <- see README
# EVERYTHING ELSE IS PINNED. Do not tune it for speed (see README).
#
# Self-healing: resumes from the newest checkpoint. Idempotent: finished work is skipped.
#
#   bash run_full_ctrl.sh                          # seed 42, both protocols  (stage 1)
#   SEEDS="43 44" bash run_full_ctrl.sh            # stage 2
#   PROTOCOLS="token" SEEDS="42 43 44" bash run_full_ctrl.sh
#   SMOKE=1 bash run_full_ctrl.sh                  # ~10 min end-to-end chain test
set -u

cd "$(dirname "$0")"
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUT="${OUT:-outputs_ctrl}"
SEEDS="${SEEDS:-42}"
PROTOCOLS="${PROTOCOLS:-token pred}"
SMOKE="${SMOKE:-0}"
MAXTRIES=5

# ---- pinned recipe (ESM-C anchor + original run). Do not edit. ----
TARGET=660e6
BATCH=32
SEQ=512
LR=6e-4
WARMUP=500
CKPT_EVERY=2500
VAL_EVERY=1000
D_MODEL=320; N_HEADS=5; N_LAYERS=30       # ESM-C: head_dim 64, depth 30
# The paper's nine matched relative depths {0,13,25,38,50,63,75,88,100}%, mapped onto
# 30 blocks (index/29, same convention as the paper's ESM-2 L0..L32 => index/32).
DEPTHS="0 4 7 11 14 18 22 26 29"
BOOT_DEPTHS="all"                         # depth labels for the bootstrap (see preset)
BOOT_STEM_PREFIX="bootstrap_h1_ctrl_esmc" # smoke uses a distinct namespace below
N_SHUFFLES=5                              # paper: "5 within-protein permutations"
# -------------------------------------------------------------------

if [ "$SMOKE" = "1" ]; then
  DATA="$HOME/own_sae_data/uniref50_smoke"; OUT="outputs_ctrl_smoke"
  SEEDS="42"; PROTOCOLS="token"; DEPTHS="0 29"
  BOOT_DEPTHS="0,100"                     # only the depths the smoke actually built
  BOOT_STEM_PREFIX="bootstrap_h1_ctrl_esmc_smoke"
  SMOKE_TRAIN="--smoke"; SMOKE_EVAL="--max-proteins 40 --sae-epochs 2"
  SMOKE_CF1="--quick --max-features 256 --min-domains 2"
  echo "### SMOKE MODE: tiny corpus, 2 depths, 40 proteins — proves the chain, not the science"
else
  SMOKE_TRAIN=""; SMOKE_EVAL=""; SMOKE_CF1=""
fi

[ -f "$DATA/tokens.npy" ] || { echo "ERROR: no corpus at $DATA. Run: $PY prep_controlled_corpus.py"; exit 1; }
[ -d cache/pdb_files ] || { echo "ERROR: no PDBs. Run: $PY fetch_pdbs.py"; exit 1; }

$PY - <<'PY' || exit 1
import sys, torch
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {dev} | torch {torch.__version__}" + (f" | {torch.cuda.get_device_name(0)}" if dev=="cuda" else ""))
if dev == "cpu": print("WARNING: no GPU — training will take days.")
import esm; print("esm anchor OK")
PY

latest_ckpt () { ls -t "$1"/model_step*.pt "$1"/model_partial.pt 2>/dev/null | head -1; }

# ---- 1. train one (objective, seed, protocol) ----
train_one () {
  local OBJ="$1" SEED="$2" PROTO="$3"
  local D; D="$DATA/$(tag "$OBJ" "$SEED" "$PROTO")"
  mkdir -p "$D"
  if [ -f "$D/model_final.pt" ]; then echo "[train $OBJ s$SEED $PROTO] done - skip"; return 0; fi
  local mp=""
  # --match-predictions ONLY affects the MLM arm (it lengthens it 1/mask_rate x).
  # CLM is identical under both protocols, so its run is shared/reused.
  [ "$PROTO" = "pred" ] && [ "$OBJ" = "mlm" ] && mp="--match-predictions"
  for try in $(seq 1 $MAXTRIES); do
    local lc res=""; lc=$(latest_ckpt "$D"); [ -n "$lc" ] && res="--resume $lc"
    echo "=== [train $OBJ s$SEED $PROTO] try $try ${mp:-token-matched} $(date) ==="
    $PY -u train_ctrl_plm.py --objective "$OBJ" --data-dir "$DATA" --out-dir "$D" \
        --seed "$SEED" --target-tokens "$TARGET" --batch-size "$BATCH" --seq-len "$SEQ" \
        --lr "$LR" --warmup "$WARMUP" --ckpt-every "$CKPT_EVERY" --val-every "$VAL_EVERY" \
        --d-model "$D_MODEL" --n-heads "$N_HEADS" --n-layers "$N_LAYERS" \
        $mp $SMOKE_TRAIN $res && return 0
    echo "[train $OBJ s$SEED $PROTO] try $try FAILED - resuming"; sleep 15
  done
  return 1
}

# CLM is protocol-independent -> one dir, reused by both protocols.
tag () { local OBJ="$1" SEED="$2" PROTO="$3"
  if [ "$OBJ" = "clm" ]; then echo "ckpt_clm_s${SEED}"; else echo "ckpt_mlm_s${SEED}_${PROTO}"; fi; }

# ---- 2+3. extract+SAE, then L_struct, for every depth ----
analyse_one () {
  local OBJ="$1" SEED="$2" PROTO="$3"
  local T; T=$(tag "$OBJ" "$SEED" "$PROTO")
  local CK="$DATA/$T/model_final.pt"
  [ -f "$CK" ] || { echo "[analyse $T] no model_final.pt - skip"; return 1; }
  for L in $DEPTHS; do
    local LD="$OUT/$T/layer_$L"
    if [ -f "$LD/struct_seq_metrics.csv" ]; then echo "[analyse $T L$L] done - skip"; continue; fi
    echo "=== [analyse $T] layer $L $(date) ==="
    $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$T" --layer "$L" \
        --out-root "$OUT" --eval-set eval_set $SMOKE_EVAL || { echo "!! eval failed $T L$L"; continue; }
    # --n-shuffles 5 is REQUIRED: cpu_stage defaults to 3, but the paper (and
    # run_all.sh:121, and compute_h1_bootstrap's N_SHUF) use 5. L_struct is
    # shuffle-corrected, so 3 would compute a different metric than the paper's.
    $PY -u cpu_stage.py --layer-dir "$LD" --model-type residue \
        --n-shuffles "$N_SHUFFLES" \
        --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
        --fasta-path cache/scope_40.fa || { echo "!! cpu_stage failed $T L$L"; continue; }
    # Concept-F1: the second, independent lens (InterPLM-style feature<->concept
    # alignment). It disagreed with L_struct on the previous pilot, so it is worth
    # having in the redo rather than resting the claim on one metric. Also reads Z.
    local CF="results_concept_f1/$T/layer_$L"
    if [ ! -f "$CF/concept_f1.csv" ]; then
      $PY -u experiment_concept_f1.py --layer-dir "$LD" --save-dir "$CF" \
          --features-csv cache/residue_features.csv --fasta-path cache/scope_40.fa \
          --split-level protein $SMOKE_CF1 || echo "!! concept_f1 failed $T L$L"
    fi
    # NOTE: Z.npy is NOT deleted here. Both compute_h1_bootstrap.py (the confidence
    # intervals) and experiment_concept_f1.py mmap Z via cpu_stage.load_layer, so it
    # must survive until those have run. prune_z() below handles cleanup.
  done
}

# ---- 4. cross-model bootstrap (the CIs). MUST run before Z is pruned. ----
bootstrap_pair () {
  local SEED="$1" PROTO="$2"
  local A B STEM
  A=$(tag mlm "$SEED" "$PROTO"); B=$(tag clm "$SEED" "$PROTO")
  STEM="${BOOT_STEM_PREFIX}_s${SEED}_${PROTO}"
  if [ -f "outputs_robustness/${STEM}_full_bylevel_minact0.csv" ]; then
    echo "[bootstrap s$SEED $PROTO] done - skip"; return 0
  fi
  echo "=== [bootstrap s$SEED $PROTO] $A vs $B $(date) ==="
  $PY -u outputs_robustness/compute_h1_bootstrap.py --preset ctrl_esmc \
      --model-a "$A" --model-b "$B" --out-stem "$STEM" --output-root "$OUT" \
      --cluster-levels fold,protein --min-active 0 --depths "$BOOT_DEPTHS" \
      --n-boot "${N_BOOT:-1000}" || echo "!! bootstrap failed s$SEED $PROTO"
}

# ---- 5. prune the big intermediates — ONLY if the bootstrap really consumed them ----
# compute_h1_bootstrap.py mmaps Z. Deleting Z when the bootstrap has NOT run would
# silently destroy the only input to the confidence intervals and force a full retrain,
# so this refuses to prune unless every expected CSV exists.
prune_z () {
  [ "${KEEP_Z:-0}" = "1" ] && { echo "KEEP_Z=1 — keeping Z.npy"; return 0; }
  local missing=0
  for s in $SEEDS; do for proto in $PROTOCOLS; do
    [ -f "outputs_robustness/${BOOT_STEM_PREFIX}_s${s}_${proto}_full_bylevel_minact0.csv" ] || missing=1
  done; done
  if [ "$missing" = "1" ]; then
    echo "REFUSING to prune Z.npy: a bootstrap CSV is missing — Z is its only input."
    echo "  Fix the bootstrap and re-run; nothing is deleted. (KEEP_Z=1 to silence.)"
    return 0
  fi
  local n; n=$(find "$OUT" -name Z.npy | wc -l | tr -d ' ')
  find "$OUT" -name Z.npy -delete
  echo "pruned $n Z.npy (~$((n*3/2)) GB) — L_struct + bootstrap already computed"
}

echo "########## controlled MLM-vs-CLM | seeds:$SEEDS | protocols:$PROTOCOLS | $(date) ##########"
for s in $SEEDS; do
  for proto in $PROTOCOLS; do
    train_one clm "$s" "$proto" || echo "!! clm s$s did not finish"
    train_one mlm "$s" "$proto" || echo "!! mlm s$s $proto did not finish"
    analyse_one clm "$s" "$proto"
    analyse_one mlm "$s" "$proto"
    bootstrap_pair "$s" "$proto"          # needs Z -> must precede prune_z
  done
done
prune_z

echo; echo "########## DONE $(date) ##########"
echo "Results to send back:"
find "$OUT" -name struct_seq_metrics.csv 2>/dev/null | sort | sed 's/^/  /'
find results_concept_f1 -name 'concept_f1.csv' 2>/dev/null | sort | sed 's/^/  /'
find outputs_robustness -name "${BOOT_STEM_PREFIX}_s*.csv" 2>/dev/null | sort | sed 's/^/  /'
echo
echo "  tar czf ctrl_results.tgz \\"
echo "    \$(find $OUT -name 'struct_seq_metrics.csv' -o -name 'META.json') \\"
echo "    results_concept_f1 outputs_robustness/${BOOT_STEM_PREFIX}_s*.csv train.log"
