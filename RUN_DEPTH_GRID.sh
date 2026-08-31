#!/usr/bin/env bash
# RUN_DEPTH_GRID.sh — widen the corpus control to the full nine-depth grid,
#                     and add the missing probe depth.
#
# For Ronnie's box. Same shape as the 2026-08-28 batch (run_gaps): resumable,
# one log per cell, packages its own results archive at the end.
#
#   bash RUN_DEPTH_GRID.sh            # both stages
#   ONLY=1 bash RUN_DEPTH_GRID.sh     # just the corpus control
#   ONLY=2 bash RUN_DEPTH_GRID.sh     # just the probes
#
# WHY
#   The corpus control (the paper's central experiment) runs at blocks 11/14/18
#   — three of thirty. The main arm effect is reported at nine depths. A reviewer
#   asks why those three. This runs the control at the other six so the two grids
#   match, which also brings blocks 7 and 22 (the frozen-dictionary depths) and
#   block 18 (probes) inside one grid.
#
# COST
#   Stage 1: 36 shuffled cells. Measured 125 s/cell on this box in the 08-28
#            batch (121-147 s across 48 cells) -> ~75 min. Native cells at the
#            same depths should already exist in outputs_ctrl/; any that don't
#            are run too, at the same cost each.
#   Stage 2: 6 arms x 1 depth x 2 probe types. Forward passes + sklearn. Well
#            under an hour, but this is an estimate, not a measurement.
#
# NOTHING HERE CHANGES THE METRIC. Same 8 A contact radius, same |i-j| >= 12
# separation floor, same top-10% active set, same 5-permutation null, same
# k=256 / expansion 8 dictionary. Only the block index moves.
set -u
cd "$(dirname "$0")"

_REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo " +local-changes")
echo "########## repo revision: ${_REV}${_DIRTY} | started $(date) ##########"

PY="${PY:-python}"
ONLY="${ONLY:-}"
SEEDS="${SEEDS:-42 43 44}"
NEW_DEPTHS="${NEW_DEPTHS:-0 4 7 22 26 29}"     # 11 14 18 already done
PROBE_DEPTH="${PROBE_DEPTH:-18}"
K_SPARSE="${K_SPARSE:-256}"
EXPANSION="${EXPANSION:-8}"
NSHUF="${NSHUF:-5}"

CKPT_ROOT="${CKPT_ROOT:-$HOME/own_sae_data/uniref50_pilot}"
CKPT_ROOT_SHUF="${CKPT_ROOT_SHUF:-$HOME/own_sae_data/uniref50_pilot_shuf}"
OUT_NATIVE="${OUT_NATIVE:-outputs_ctrl}"
OUT_SHUF="${OUT_SHUF:-outputs_ctrl_shuf}"

LOGDIR="logs_depthgrid"; mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FAILED=""

arms_for_seed () { echo "ckpt_mlm_s${1}_token ckpt_clm_s${1}"; }

# ------------------------------------------------------------------ preflight
# Check 1 exists because the failure it catches produces PLAUSIBLE NUMBERS
# rather than an error: if the shuffled root resolves to the real checkpoints,
# the control compares the real models against themselves and reports
# "shuffled == real", which reads as the metric PASSING a validity check.
echo "=== preflight ==="
_bad=0
for s in $SEEDS; do
  for arm in $(arms_for_seed "$s"); do
    for pair in "REAL:$CKPT_ROOT" "SHUF:$CKPT_ROOT_SHUF"; do
      tag="${pair%%:*}"; root="${pair##*:}"
      ck="$root/$arm/model_final.pt"
      [ -f "$ck" ] || { echo "  !! MISSING $tag $ck"; _bad=1; }
    done
    a="$CKPT_ROOT/$arm/model_final.pt"; b="$CKPT_ROOT_SHUF/$arm/model_final.pt"
    if [ -f "$a" ] && [ -f "$b" ]; then
      ha=$(sha256sum "$a" | cut -d' ' -f1); hb=$(sha256sum "$b" | cut -d' ' -f1)
      if [ "$ha" = "$hb" ]; then
        echo "  !! IDENTICAL WEIGHTS: $arm real and shuffled are the same file."
        echo "     Fix CKPT_ROOT_SHUF before running. Do not proceed."
        _bad=1
      fi
    fi
  done
done
[ "$_bad" -eq 0 ] && echo "  checkpoints OK, real and shuffled distinct"
[ -d "$OUT_NATIVE" ] || echo "  NOTE: no $OUT_NATIVE/ — native cells will be built from scratch (slower)"
[ "$_bad" -ne 0 ] && { echo "preflight FAILED — stopping"; exit 1; }
echo

want () { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# ---------------------------------------------------------- one (arm, depth) cell
run_cell () {   # run_cell <ckpt_root> <out_root> <arm> <layer>
  local root="$1" out="$2" arm="$3" L="$4"
  local ck="$root/$arm/model_final.pt"
  local dest="$out/$arm/layer_$L"
  [ -f "$ck" ] || { echo "  [skip] no $ck"; return 0; }
  if [ -f "$dest/struct_seq_metrics.csv" ]; then echo "  [done] $out $arm L$L"; return 0; fi
  echo "=== [$out] $arm L$L $(date +%H:%M:%S) ==="
  local log="$LOGDIR/cell_${out}_${arm}_L${L}_${STAMP}.log"
  {
    $PY -u eval_ctrl_plm.py --ckpt "$ck" --name "$arm" --layer "$L" \
        --out-root "$out" --eval-set eval_set \
        --k-sparse "$K_SPARSE" --expansion "$EXPANSION" &&
    $PY -u cpu_stage.py --layer-dir "$dest" --model-type residue --n-shuffles "$NSHUF"
  } > "$log" 2>&1 || { echo "  !! FAILED (see $log)"; FAILED="$FAILED ${out}/${arm}/L${L}"; return 1; }
  grep -E "val_EV|DEGENERATE" "$log" | tail -1 | sed 's/^/    /'
  return 0
}

# ------------------------------------------------------------------ stage 1
if want 1; then
  echo "============================================================"
  echo "  STAGE 1  corpus control at the six missing depths"
  echo "  depths: $NEW_DEPTHS   seeds: $SEEDS"
  echo "============================================================"
  for L in $NEW_DEPTHS; do
    for s in $SEEDS; do
      for arm in $(arms_for_seed "$s"); do
        run_cell "$CKPT_ROOT_SHUF" "$OUT_SHUF"   "$arm" "$L"
        run_cell "$CKPT_ROOT"      "$OUT_NATIVE" "$arm" "$L"
      done
    done
  done
  echo "  [STAGE 1] complete"
fi

# ------------------------------------------------------------------ stage 2
# Probes currently run at 7/11/14. Block 18 is the deepest corpus-control depth
# and the one where the masked L_struct advantage has already halved, so it is
# where the 27/27 causal-favouring result is most likely to change.
if want 2; then
  echo
  echo "============================================================"
  echo "  STAGE 2  SAE-free probes at block $PROBE_DEPTH"
  echo "============================================================"
  ALL_ARMS=""
  for s in $SEEDS; do
    for arm in $(arms_for_seed "$s"); do
      [ -f "$CKPT_ROOT/$arm/model_final.pt" ] && ALL_ARMS="${ALL_ARMS:+$ALL_ARMS,}$arm"
    done
  done
  if [ -z "$ALL_ARMS" ]; then
    echo "  !! no native checkpoints found — skipping"; FAILED="$FAILED probes"
  else
    for variant in linear mlp; do
      out="results_ctrl_saefree_L${PROBE_DEPTH}"; flag=""
      [ "$variant" = "mlp" ] && { out="${out}_mlp"; flag="--mlp"; }
      if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
        echo "  [done] $variant ($out non-empty)"; continue
      fi
      echo "=== [probe $variant] L$PROBE_DEPTH $(date +%H:%M:%S) ==="
      log="$LOGDIR/probe_${variant}_L${PROBE_DEPTH}_${STAMP}.log"
      $PY -u eval_ctrl_saefree.py --ckpt-root "$CKPT_ROOT" --arms "$ALL_ARMS" \
          --depths "$PROBE_DEPTH" --eval-set eval_set --out "$out" $flag \
          > "$log" 2>&1 || { echo "  !! FAILED (see $log)"; FAILED="$FAILED probe-$variant"; continue; }
      tail -5 "$log" | sed 's/^/    /'
    done
  fi
  echo "  [STAGE 2] complete"
fi

# ------------------------------------------------------------------ package
echo
echo "============================================================"
echo "  packaging"
echo "============================================================"
PKG="depthgrid_results_$(date +%Y%m%d)"
rm -rf "$PKG"; mkdir -p "$PKG"
for out in "$OUT_SHUF" "$OUT_NATIVE"; do
  [ -d "$out" ] || continue
  find "$out" -name struct_seq_metrics.csv -print0 2>/dev/null | while IFS= read -r -d '' f; do
    mkdir -p "$PKG/$(dirname "$f")"; cp "$f" "$PKG/$f"
  done
done
for d in results_ctrl_saefree_L${PROBE_DEPTH} results_ctrl_saefree_L${PROBE_DEPTH}_mlp; do
  [ -d "$d" ] && cp -r "$d" "$PKG/"
done
mkdir -p "$PKG/logs"; cp "$LOGDIR"/*_"$STAMP".log "$PKG/logs/" 2>/dev/null
git rev-parse HEAD > "$PKG/git_revision.txt" 2>/dev/null
git status --short > "$PKG/git_status.txt" 2>/dev/null
( cd "$PKG" && find . -type f ! -name SHA256SUMS.txt -exec sha256sum {} + > SHA256SUMS.txt )

{
  echo "# Depth-grid batch"
  echo
  echo "- Date: $(date +%Y-%m-%d)"
  echo "- Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "- New depths: $NEW_DEPTHS (11/14/18 already existed)"
  echo "- Seeds: $SEEDS"
  echo "- Metric settings: 8 A, |i-j| >= 12, top-10%, $NSHUF permutations (unchanged)"
  echo
  echo "## Mean L_struct per cell (struct_delta averaged over features)"
  echo
  echo "| condition | arm | layer | mean L_struct |"
  echo "|---|---|---:|---:|"
  for out in "$OUT_SHUF" "$OUT_NATIVE"; do
    [ -d "$out" ] || continue
    find "$out" -name struct_seq_metrics.csv 2>/dev/null | sort | while read -r f; do
      arm=$(echo "$f" | awk -F/ '{print $2}'); lay=$(echo "$f" | awk -F/ '{print $3}')
      m=$($PY - "$f" <<'PYEOF'
import csv, sys, statistics as st
with open(sys.argv[1]) as fh:
    print(f"{st.mean(float(r['struct_delta']) for r in csv.DictReader(fh)):+.5f}")
PYEOF
)
      echo "| $out | $arm | ${lay#layer_} | $m |"
    done
  done
  [ -n "$FAILED" ] && { echo; echo "## FAILED cells"; echo; echo "$FAILED"; }
} > "$PKG/RESULT_SUMMARY.md"

tar czf "${PKG}.tgz" "$PKG"
echo "  wrote ${PKG}.tgz  ($(du -h "${PKG}.tgz" | cut -f1))"
echo
if [ -n "$FAILED" ]; then
  echo "FAILURES:$FAILED"
  echo "Send the archive anyway — partial cells are usable."
  exit 1
fi
echo "All cells OK. Send ${PKG}.tgz, plus the two items in HANDOFF.md 'Asks for Ronnie'."
