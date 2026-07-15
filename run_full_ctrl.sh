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
DEPTHS="0 3 7 10 14 17 21 25 29"          # 9 matched relative depths over 30 blocks
# -------------------------------------------------------------------

if [ "$SMOKE" = "1" ]; then
  DATA="$HOME/own_sae_data/uniref50_smoke"; OUT="outputs_ctrl_smoke"
  SEEDS="42"; PROTOCOLS="token"; DEPTHS="0 29"
  SMOKE_TRAIN="--smoke"; SMOKE_EVAL="--max-proteins 40 --sae-epochs 2"
  echo "### SMOKE MODE: tiny corpus, 2 depths, 40 proteins — proves the chain, not the science"
else
  SMOKE_TRAIN=""; SMOKE_EVAL=""
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
    $PY -u cpu_stage.py --layer-dir "$LD" --model-type residue \
        --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
        --fasta-path cache/scope_40.fa || { echo "!! cpu_stage failed $T L$L"; continue; }
    # Z.npy is ~1.5 GB/layer and is not needed once L_struct exists.
    [ "${KEEP_Z:-0}" = "1" ] || rm -f "$LD/Z.npy"
  done
}

echo "########## controlled MLM-vs-CLM | seeds:$SEEDS | protocols:$PROTOCOLS | $(date) ##########"
for s in $SEEDS; do
  for proto in $PROTOCOLS; do
    train_one clm "$s" "$proto" || echo "!! clm s$s did not finish"
    train_one mlm "$s" "$proto" || echo "!! mlm s$s $proto did not finish"
    analyse_one clm "$s" "$proto"
    analyse_one mlm "$s" "$proto"
  done
done

echo; echo "########## DONE $(date) ##########"
echo "Send back these (small) files:"
find "$OUT" -name struct_seq_metrics.csv | sort | sed 's/^/  /'
echo
echo "  tar czf ctrl_results.tgz \$(find $OUT -name 'struct_seq_metrics.csv' -o -name 'META.json') train.log"
