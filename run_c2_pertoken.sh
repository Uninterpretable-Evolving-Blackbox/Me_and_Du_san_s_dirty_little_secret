#!/usr/bin/env bash
# run_c2_pertoken.sh — C2: redo the controlled MLM-vs-CLM comparison with the SAE
# INSTRUMENT FIX (per-token normalisation), on the checkpoints you already trained.
#
# WHY: the validity check (run_validity_check.sh) showed the MLM SAEs are degenerate
# (val_EV >= 0.99) at the shallowest depths — so the raw MLM-vs-CLM L_struct comparison
# there is not trustworthy. This re-runs the ANALYSIS ONLY (no re-training): extract ->
# per-token-normalise -> SAE -> L_struct + val_EV, for both arms at all 9 depths, and
# also the un-normalised BASELINE (if not already present) so Wei can plot before/after.
# This is the controlled-experiment version of the ESM-2/RITA "A2" fix.
#
# COST: analysis only (~3-5 min/depth on the RTX: a forward pass + one SAE + L_struct).
# ~9 depths x 3 arms x up-to-2 conditions ~= a few hours; resumable & idempotent.
#
# PREREQ: the trained checkpoints must still exist under $DATA (do NOT delete
# ~/own_sae_data), plus cache/pdb_files (from fetch_pdbs.py) for L_struct.
#
#   bash run_c2_pertoken.sh                 # both arms, both protocols, 9 depths
#   ARMS="clm mlm_token" bash run_c2_pertoken.sh   # headline pair only (faster)
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUT="${OUT:-outputs_ctrl}"
ARMS="${ARMS:-clm mlm_token mlm_pred}"
DEPTHS="${DEPTHS:-0 4 7 11 14 18 22 26 29}"   # 9 paper-matched depths
DO_BASELINE="${DO_BASELINE:-1}"               # also (re)make un-normalised baseline if missing

ckpt_dir () {  # arm -> checkpoint dir name (matches run_full_ctrl.sh's tag())
  case "$1" in clm) echo "ckpt_clm_s42";; mlm_token) echo "ckpt_mlm_s42_token";;
               mlm_pred) echo "ckpt_mlm_s42_pred";; *) echo "ckpt_$1";; esac
}

[ -d cache/pdb_files ] || { echo "ERROR: no cache/pdb_files (run fetch_pdbs.py). L_struct needs it."; exit 1; }
$PY - <<'PY' || exit 1
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow)")
PY

analyse () {   # arm, layer, condition(base|ptn)
  local ARM="$1"
  local L="$2"
  local COND="$3"
  local CK; CK="$DATA/$(ckpt_dir "$ARM")/model_final.pt"
  [ -f "$CK" ] || { echo "  [skip] $ARM L$L $COND — no $CK"; return 0; }
  local NAME="ctrl_${ARM}$([ "$COND" = ptn ] && echo _ptn)"
  local LD="$OUT/$NAME/layer_$L"
  local ptnflag=""; [ "$COND" = ptn ] && ptnflag="--per-token-norm"
  if [ -f "$LD/struct_seq_metrics.csv" ]; then echo "  [done] $NAME L$L"; return 0; fi
  echo "=== $NAME L$L ($COND) $(date +%H:%M:%S) ==="
  $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$NAME" --layer "$L" \
      --out-root "$OUT" --eval-set eval_set $ptnflag || { echo "!! eval $NAME L$L"; return 1; }
  $PY -u cpu_stage.py --layer-dir "$LD" --model-type residue --n-shuffles 5 \
      --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
      --fasta-path cache/scope_40.fa || { echo "!! cpu_stage $NAME L$L"; return 1; }
  [ "${KEEP_Z:-0}" = "1" ] || rm -f "$LD/Z.npy"   # L_struct done; Z not needed further
}

echo "########## C2 per-token-norm | arms:$ARMS | $(date) ##########"
for arm in $ARMS; do
  for L in $DEPTHS; do
    [ "$DO_BASELINE" = "1" ] && analyse "$arm" "$L" base
    analyse "$arm" "$L" ptn
  done
done

echo; echo "########## C2 DONE $(date) ##########"
echo "Send back (small):"
find "$OUT" -path "*ctrl_*" -name struct_seq_metrics.csv 2>/dev/null | sort | sed 's/^/  /'
echo
echo "  tar czf c2_results.tgz \$(find $OUT -name 'struct_seq_metrics.csv' -o -name 'META.json' | grep ctrl_)"
