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
# The full chain, matching run_full_ctrl.sh (a point estimate on its own is not a result):
#   extract+SAE -> L_struct -> concept-F1 -> fold-cluster bootstrap CIs -> prune Z
# Both concept-F1 and the bootstrap mmap Z.npy, so Z survives until they have run;
# prune_z() refuses to delete it otherwise. Skipping either step would leave C2 with
# point-estimate Cohen's d only — no CIs (the project's consistency rule) and no
# second lens (which is what C4 reconciles).
#
# COST: analysis only (~3-5 min/depth on the RTX: a forward pass + one SAE + L_struct),
# plus ~2-5 min/depth for concept-F1 and ~30 min per bootstrap pair (B=1000, rebuilds
# the Ca adjacency once). ~9 depths x 3 arms x up-to-2 conditions ~= a few hours, then
# up to 4 bootstrap pairs. Resumable & idempotent throughout — interrupt freely.
#
# PREREQ: the trained checkpoints must still exist under $DATA (do NOT delete
# ~/own_sae_data), plus cache/pdb_files (from fetch_pdbs.py) for L_struct.
#
#   bash run_c2_pertoken.sh                 # both arms, both protocols, 9 depths
#   ARMS="clm mlm_token" bash run_c2_pertoken.sh   # headline pair only (faster)
#   DO_CF1=0 bash run_c2_pertoken.sh        # skip concept-F1 (leaves C4 unanswerable)
#   N_BOOT=200 bash run_c2_pertoken.sh      # quicker, wider CIs
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
DATA="${DATA:-$HOME/own_sae_data/uniref50_pilot}"
OUT="${OUT:-outputs_ctrl}"
ARMS="${ARMS:-clm mlm_token mlm_pred}"
DEPTHS="${DEPTHS:-0 4 7 11 14 18 22 26 29}"   # 9 paper-matched depths
DO_BASELINE="${DO_BASELINE:-1}"               # also (re)make un-normalised baseline if missing
DO_CF1="${DO_CF1:-1}"                         # concept-F1, the second lens (C4)
DO_BOOT="${DO_BOOT:-1}"                       # fold-cluster bootstrap CIs
N_BOOT="${N_BOOT:-1000}"                      # paper setting
BOOT_STEM_PREFIX="${BOOT_STEM_PREFIX:-bootstrap_h1_c2}"

ckpt_dir () {  # arm -> checkpoint dir name (matches run_full_ctrl.sh's tag())
  case "$1" in clm) echo "ckpt_clm_s42";; mlm_token) echo "ckpt_mlm_s42_token";;
               mlm_pred) echo "ckpt_mlm_s42_pred";; *) echo "ckpt_$1";; esac
}

# The bootstrap selects depths by LABEL ({0,13,...,100}%), not by block index, so a
# DEPTHS override has to be translated or the CIs would silently cover the wrong set.
boot_depths () {
  local out=""
  local l
  for l in $DEPTHS; do
    case "$l" in
      0) out="$out,0";;   4) out="$out,13";;  7) out="$out,25";;
      11) out="$out,38";; 14) out="$out,50";; 18) out="$out,63";;
      22) out="$out,75";; 26) out="$out,88";; 29) out="$out,100";;
      *) echo "WARN: block $l has no depth label — excluded from the bootstrap" >&2;;
    esac
  done
  echo "${out#,}"
}

[ -d cache/pdb_files ] || { echo "ERROR: no cache/pdb_files (run fetch_pdbs.py). L_struct needs it."; exit 1; }
$PY - <<'PY' || exit 1
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow)")
PY

# ---- 1+2. extract+SAE -> L_struct -> concept-F1, per (arm, depth, condition) ----
analyse () {   # arm, layer, condition(base|ptn)
  local ARM="$1"
  local L="$2"
  local COND="$3"
  local CK; CK="$DATA/$(ckpt_dir "$ARM")/model_final.pt"
  [ -f "$CK" ] || { echo "  [skip] $ARM L$L $COND — no $CK"; return 0; }
  local NAME="ctrl_${ARM}$([ "$COND" = ptn ] && echo _ptn)"
  local LD="$OUT/$NAME/layer_$L"
  local ptnflag=""; [ "$COND" = ptn ] && ptnflag="--per-token-norm"

  if [ -f "$LD/struct_seq_metrics.csv" ]; then
    echo "  [done] $NAME L$L (L_struct)"
  else
    echo "=== $NAME L$L ($COND) $(date +%H:%M:%S) ==="
    $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$NAME" --layer "$L" \
        --out-root "$OUT" --eval-set eval_set $ptnflag || { echo "!! eval $NAME L$L"; return 1; }
    # --n-shuffles 5 is REQUIRED: cpu_stage defaults to 3, but the paper uses 5, and
    # L_struct is shuffle-corrected — 3 would compute a different metric than the paper's.
    $PY -u cpu_stage.py --layer-dir "$LD" --model-type residue --n-shuffles 5 \
        --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
        --fasta-path cache/scope_40.fa || { echo "!! cpu_stage $NAME L$L"; return 1; }
  fi

  # Concept-F1: the second, independent lens (InterPLM-style feature<->concept alignment).
  # It disagreed with L_struct on the previous pilot, so C4 asks whether the instrument fix
  # resolves that. Reads Z via cpu_stage.load_layer -> must precede prune_z().
  local CF="results_concept_f1/$NAME/layer_$L"
  if [ "$DO_CF1" = "1" ] && [ ! -f "$CF/concept_f1.csv" ]; then
    if [ -f "$LD/Z.npy" ]; then
      $PY -u experiment_concept_f1.py --layer-dir "$LD" --save-dir "$CF" \
          --features-csv cache/residue_features.csv --fasta-path cache/scope_40.fa \
          --split-level protein || echo "!! concept_f1 $NAME L$L"
    else
      echo "  !! $NAME L$L: Z.npy is gone (an older revision of this script deleted it)."
      echo "     To restore: rm -rf $LD and re-run — extract+SAE+L_struct will redo it."
    fi
  fi
}

# ---- 3. fold-cluster bootstrap CIs, per (mlm arm, condition). MUST precede prune_z. ----
bootstrap_pair () {   # mlm-arm, condition(base|ptn)
  local ARM="$1"
  local COND="$2"
  local SFX=""; [ "$COND" = ptn ] && SFX="_ptn"
  local A="ctrl_${ARM}${SFX}"
  local B="ctrl_clm${SFX}"
  local STEM="${BOOT_STEM_PREFIX}_${ARM}_${COND}"
  if [ -f "outputs_robustness/${STEM}_full_bylevel_minact0.csv" ]; then
    echo "[bootstrap $ARM $COND] done - skip"; return 0
  fi
  [ -d "$OUT/$A" ] && [ -d "$OUT/$B" ] || {
    echo "[bootstrap $ARM $COND] $A or $B missing - skip"; return 0; }
  echo "=== [bootstrap $ARM $COND] $A vs $B $(date) ==="
  $PY -u outputs_robustness/compute_h1_bootstrap.py --preset ctrl_esmc \
      --model-a "$A" --model-b "$B" --out-stem "$STEM" --output-root "$OUT" \
      --cluster-levels fold,protein --min-active 0 --depths "$(boot_depths)" \
      --n-boot "$N_BOOT" || echo "!! bootstrap $ARM $COND"
}

# ---- 4. prune the big intermediates — ONLY once the CIs really consumed them ----
# Deleting Z when a bootstrap has NOT run silently destroys the only input to the
# confidence intervals and forces a full re-extraction, so this refuses unless every
# expected CSV exists (same guard as run_full_ctrl.sh).
prune_z () {
  [ "${KEEP_Z:-0}" = "1" ] && { echo "KEEP_Z=1 — keeping Z.npy"; return 0; }
  if [ "$DO_BOOT" != "1" ]; then
    echo "DO_BOOT=0 — keeping Z.npy (the bootstrap still needs it)"; return 0
  fi
  local missing=0
  local arm cond
  for arm in $ARMS; do
    [ "$arm" = "clm" ] && continue
    for cond in base ptn; do
      [ "$cond" = "base" ] && [ "$DO_BASELINE" != "1" ] && continue
      [ -f "outputs_robustness/${BOOT_STEM_PREFIX}_${arm}_${cond}_full_bylevel_minact0.csv" ] || missing=1
    done
  done
  if [ "$missing" = "1" ]; then
    echo "REFUSING to prune Z.npy: a bootstrap CSV is missing — Z is its only input."
    echo "  Fix the bootstrap and re-run; nothing is deleted. (KEEP_Z=1 to silence.)"
    return 0
  fi
  # Delete ONLY the arms this invocation actually verified. The guard above loops over $ARMS,
  # so a narrowed run (e.g. ARMS="clm mlm_token", which this script's own header suggests) would
  # otherwise delete Z.npy for mlm_pred — whose CIs were never computed and cannot be recomputed,
  # because analyse() sees struct_seq_metrics.csv and skips the extraction that would recreate Z.
  local n=0 arm cond sfx d
  for arm in $ARMS; do
    for cond in base ptn; do
      [ "$cond" = "base" ] && [ "$DO_BASELINE" != "1" ] && continue
      sfx=""; [ "$cond" = ptn ] && sfx="_ptn"
      d="$OUT/ctrl_${arm}${sfx}"
      [ -d "$d" ] || continue
      n=$((n + $(find "$d" -name Z.npy | wc -l | tr -d ' ')))
      find "$d" -name Z.npy -delete
    done
  done
  echo "pruned $n Z.npy across arms [$ARMS] — L_struct, concept-F1 and bootstrap already computed"
  local left; left=$(find "$OUT" -path "*ctrl_*" -name Z.npy | wc -l | tr -d ' ')
  [ "$left" != "0" ] && echo "  kept $left Z.npy belonging to arms this run did not verify"
}

echo "########## C2 per-token-norm | arms:$ARMS | $(date) ##########"
for arm in $ARMS; do
  for L in $DEPTHS; do
    [ "$DO_BASELINE" = "1" ] && analyse "$arm" "$L" base
    analyse "$arm" "$L" ptn
  done
done

# Bootstraps run after ALL arms are analysed: each pair needs its MLM arm and the
# shared CLM arm complete.
if [ "$DO_BOOT" = "1" ]; then
  echo; echo "########## C2 BOOTSTRAP CIs $(date) ##########"
  for arm in $ARMS; do
    [ "$arm" = "clm" ] && continue
    [ "$DO_BASELINE" = "1" ] && bootstrap_pair "$arm" base
    bootstrap_pair "$arm" ptn
  done
fi

prune_z

echo; echo "########## C2 DONE $(date) ##########"
echo "Send back (small):"
find "$OUT" -path "*ctrl_*" -name struct_seq_metrics.csv 2>/dev/null | sort | sed 's/^/  /'
find results_concept_f1 -path "*ctrl_*" -name 'concept_f1.csv' 2>/dev/null | sort | sed 's/^/  /'
find outputs_robustness -name "${BOOT_STEM_PREFIX}_*.csv" 2>/dev/null | sort | sed 's/^/  /'
echo
echo "  tar czf c2_results.tgz \\"
echo "    \$(find $OUT -name 'struct_seq_metrics.csv' -o -name 'META.json' | grep ctrl_) \\"
echo "    results_concept_f1 outputs_robustness/${BOOT_STEM_PREFIX}_*.csv"
