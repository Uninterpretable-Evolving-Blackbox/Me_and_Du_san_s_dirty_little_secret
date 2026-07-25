#!/usr/bin/env bash
# run_crosscoder_ctrl.sh — the MLM-vs-CLM model-diffing crosscoder, end to end.
#
# WHAT IT ANSWERS
#   Every result so far trained a SEPARATE dictionary per arm, so "this feature exists under one
#   objective and not the other" was not expressible. This trains ONE shared dictionary with
#   per-arm decoders and asks: are the structurally-local latents SHARED between the masked and
#   causal models, or UNIQUE to the masked one?
#
# WHY HERE AND NOT ON WEI'S MAC
#   Both arms are the same architecture (d=320, one backbone, one init, only the objective
#   differs), which is what makes the field's validity diagnostic (Latent Scaling) applicable at
#   all — it needs equal widths. On ESM-2 (1280) vs RITA (1536) it is undefined. So this pair is
#   the headline; the Mac runs the cross-architecture extension.
#
# ANALYSIS ONLY — no training. Reuses the checkpoints you already have.
#
# COST: ~10-25 min per (depth, seed) on the RTX — one forward pass per arm, one crosscoder,
# one L_struct. Resumable & idempotent: finished cells are skipped.
#   default = 3 depths x 3 seeds ~= 2-4 h. ALL=1 does all 9 depths at seed 42 as well.
#
# PREREQ: checkpoints under $DATA (do NOT delete ~/own_sae_data) + cache/pdb_files.
#
#   bash run_crosscoder_ctrl.sh                    # 3 depths x seeds 42,43,44
#   ARM_A=mlm_pred bash run_crosscoder_ctrl.sh     # prediction-matched protocol instead
#   ALL=1 bash run_crosscoder_ctrl.sh              # + full 9-depth profile at seed 42
#   DEPTHS="14" SEEDS="42" bash run_crosscoder_ctrl.sh   # single quick cell
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUT="${OUT:-outputs_crosscoder}"
ARM_A="${ARM_A:-mlm_token}"      # reference arm (sets hidden/k)
ARM_B="${ARM_B:-clm}"
SEEDS="${SEEDS:-42 43 44}"       # crosscoder init seeds — the project standard is 3, not 5
DEPTHS="${DEPTHS:-7 14 22}"      # 25% / 50% / 75%; C1 showed 0 and 4 are the degenerate ones
ALL="${ALL:-0}"
ALL_DEPTHS="0 4 7 11 14 18 22 26 29"

ckpt_dir () {  # arm -> checkpoint dir (matches run_full_ctrl.sh's tag())
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

NAME_A="${ARM_A}"; NAME_B="${ARM_B}"

cell () {  # layer, seed
  local L="$1"
  local S="$2"
  local D="$OUT/cc_${NAME_A}_${NAME_B}_l${L}_seed${S}"
  if [ -f "$D/struct_seq_metrics.csv" ] && grep -q spearman_dn_lstruct "$D/META.json" 2>/dev/null; then
    echo "  [done] L$L seed $S"; return 0
  fi
  echo "=== crosscoder ${NAME_A} vs ${NAME_B} | L$L | seed $S | $(date +%H:%M:%S) ==="
  if [ ! -f "$D/latent_diffing.csv" ]; then
    $PY -u eval_crosscoder_ctrl.py --ckpt-a "$CKA" --ckpt-b "$CKB" \
        --name-a "$NAME_A" --name-b "$NAME_B" --layer "$L" --seed "$S" \
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

echo "########## CROSSCODER ${NAME_A} vs ${NAME_B} | depths:$DEPTHS | seeds:$SEEDS | $(date) ##########"
for L in $DEPTHS; do for S in $SEEDS; do cell "$L" "$S"; done; done

if [ "$ALL" = "1" ]; then
  echo; echo "########## depth profile, seed 42 $(date) ##########"
  for L in $ALL_DEPTHS; do cell "$L" 42; done
fi

echo; echo "########## DONE $(date) ##########"
echo "Send back (small — no Z):"
find "$OUT" -name 'latent_diffing.csv' -o -name 'META.json' -o -name 'struct_seq_metrics.csv' \
  2>/dev/null | sort | sed 's/^/  /'
echo
echo "  tar czf crosscoder_results.tgz \$(find $OUT -name 'latent_diffing.csv' -o -name 'META.json' -o -name 'struct_seq_metrics.csv')"
