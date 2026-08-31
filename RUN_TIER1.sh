#!/usr/bin/env bash
# RUN_TIER1.sh — the pre-submission queue that needs no pretraining.
#
# For Ronnie's box. Same shape as RUN_DEPTH_GRID.sh: resumable, one log per
# cell, an empty stage is a FAILURE not a green tick, and it packages its own
# archive at the end.
#
#   bash RUN_TIER1.sh              # all four stages
#   ONLY=1 bash RUN_TIER1.sh       # just one
#   bash RUN_TIER1.sh --plan       # print the plan and the cost, run nothing
#
# WHAT IS AND IS NOT HERE
#   Four items from the Tier 1/2 list need code that did not exist. They are
#   stages 1-4 below. Four more were already scripted or already run; this
#   script does NOT duplicate them, it prints the exact command at the end.
#   Three were excluded as too expensive to fit before the deadline; see
#   "NOT IN THIS SCRIPT" at the bottom.
#
# NOTHING HERE RETRAINS A MODEL OR REFITS A DICTIONARY.
#   Stages 1-3 are CPU only. Stage 4 needs the GPU for a forward pass only.
#   cpu_stage.py is not modified: stage 2 imports from it.
set -u

cd "$(dirname "$0")"

PY="${PY:-python}"
ONLY="${ONLY:-}"
PLAN=0
[ "${1:-}" = "--plan" ] && PLAN=1

SEEDS="${SEEDS:-42 43 44}"
DEPTHS="${DEPTHS:-11 14 18}"
NSHUF="${NSHUF:-5}"          # MUST match the reference run: cpu_stage defaults to 3
K_SPARSE="${K_SPARSE:-256}"
EXPANSION="${EXPANSION:-8}"
GATES="${GATES:-global raw}"
DENOMS="${DENOMS:-sd,fixed,iqr,rank}"

CKPT_ROOT="${CKPT_ROOT:-$HOME/own_sae_data/uniref50_pilot}"
CKPT_ROOT_SHUF="${CKPT_ROOT_SHUF:-$HOME/own_sae_data/uniref50_pilot_shuf}"
OUT_NATIVE="${OUT_NATIVE:-outputs_ctrl}"
OUT_SHUF="${OUT_SHUF:-outputs_ctrl_shuf}"

LOGDIR="logs_tier1"; mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FAILED=""
RAN=0

_REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo " +local-changes")

arms_for_seed () { echo "ckpt_mlm_s${1}_token ckpt_clm_s${1}"; }
want () { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
banner () { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }

# --------------------------------------------------------------- plan / cost
if [ "$PLAN" = 1 ]; then
  cat <<'PLANEOF'
STAGE 1  top1_share agreement                 CPU     seconds
         Reads results_rigor/aa_selectivity.csv. No compute. Says whether
         Section 3's composition check survives its own robustness measure.

STAGE 2  fixed / rank denominator rescoring    CPU     ~2.5x one cpu_stage pass
         18 headline cells. Recomputes L_struct with the SD denominator
         replaced. Self-checks against struct_seq_metrics.csv first and
         refuses to write if it does not reproduce it exactly.

STAGE 3  d_struct at three model seeds         CPU     ~10 min/cell
         2 roots x 2 arms x 3 seeds x 3 depths x 2 gates = 72 cells.
         Currently the d_struct control is one seed. GATES=global halves
         this to 36 cells if the gate comparison is not being extended.

STAGE 4  d_struct untrained baseline           GPU+CPU ~10 min/cell + embed
         3 init seeds x 2 arms x 3 depths = 18 cells. The only "not run"
         cell in Table 4.
PLANEOF
  echo
  echo "Set ONLY=<n> to run one stage. SEEDS/DEPTHS/GATES/DENOMS override the grid."
  exit 0
fi

echo "########## RUN_TIER1 | rev ${_REV}${_DIRTY} | started $(date) ##########"
echo "  seeds: $SEEDS | depths: $DEPTHS | n_shuffles: $NSHUF"

# ------------------------------------------------------------------ preflight
banner "preflight"
_bad=0
for f in rescore_denominator.py analyze_top1_agreement.py cpu_stage.py \
         experiment_interplm_metric.py eval_ctrl_plm.py; do
  [ -f "$f" ] || { echo "  !! missing $f"; _bad=1; }
done

# stage 2 imports cpu_stage; if that import fails nothing downstream is
# comparable to the published numbers, and it fails 40 minutes in otherwise.
if want 2; then
  if $PY -c "import cpu_stage" >/dev/null 2>&1; then
    echo "  cpu_stage imports OK"
  else
    echo "  !! 'import cpu_stage' fails under $PY — stage 2 cannot run"
    $PY -c "import cpu_stage" 2>&1 | tail -3 | sed 's/^/     /'
    _bad=1
  fi
fi

# count the cells each stage can actually see, so an empty stage is caught here
n_z=0
for root in "$OUT_NATIVE" "$OUT_SHUF"; do
  [ -d "$root" ] || continue
  for s in $SEEDS; do for arm in $(arms_for_seed "$s"); do for L in $DEPTHS; do
    [ -f "$root/$arm/layer_$L/Z.npy" ] && n_z=$((n_z+1))
  done; done; done
done
echo "  layer dirs with Z.npy in scope: $n_z"
if { want 2 || want 3; } && [ "$n_z" -eq 0 ]; then
  echo "  !! stages 2 and 3 both need Z.npy and there is none."
  echo "     Rebuild with: STAGE=1 bash RUN_MUSTRUNS.sh   (~2 h)"
  _bad=1
fi
if want 4; then
  for s in $SEEDS; do for arm in $(arms_for_seed 42); do
    [ -f "$CKPT_ROOT/$arm/model_final.pt" ] || { echo "  !! missing $CKPT_ROOT/$arm/model_final.pt"; _bad=1; }
  done; done
fi
[ "$_bad" -ne 0 ] && { echo "preflight FAILED — stopping, nothing run"; exit 1; }
echo "  preflight OK"

# =========================================================== STAGE 1  top1
if want 1; then
  banner "STAGE 1  top1_share agreement (CPU, seconds)"
  log="$LOGDIR/s1_top1_${STAMP}.log"
  if $PY -u analyze_top1_agreement.py > "$log" 2>&1; then
    grep -E "VERDICT|rho_|check count|Section 3" "$log" | sed 's/^/  /'
    RAN=$((RAN+1))
  else
    echo "  !! FAILED (see $log)"; tail -5 "$log" | sed 's/^/     /'
    FAILED="$FAILED stage1-top1"
  fi
fi

# =========================================================== STAGE 2  denominator
if want 2; then
  banner "STAGE 2  fixed / rank denominator rescoring (CPU)"
  echo "  denominators: $DENOMS"
  echo "  each cell self-checks against struct_seq_metrics.csv before writing"
  n=0
  for root in "$OUT_NATIVE" "$OUT_SHUF"; do
    [ -d "$root" ] || { echo "  [skip] no $root/"; continue; }
    for s in $SEEDS; do for arm in $(arms_for_seed "$s"); do for L in $DEPTHS; do
      d="$root/$arm/layer_$L"
      [ -f "$d/Z.npy" ] || { echo "  [skip] no Z.npy in $d"; continue; }
      [ -f "$d/struct_seq_metrics.csv" ] || { echo "  [skip] no reference CSV in $d"; continue; }
      if [ -f "$d/struct_seq_metrics_denominators.csv" ]; then
        echo "  [done] $d"; n=$((n+1)); continue
      fi
      echo "=== [denom] $d $(date +%H:%M:%S) ==="
      log="$LOGDIR/s2_denom_$(echo "$d" | tr '/' '_')_${STAMP}.log"
      if $PY -u rescore_denominator.py --layer-dir "$d" \
             --denominators "$DENOMS" --n-shuffles "$NSHUF" > "$log" 2>&1; then
        grep -E "self-check|^    (sd|fixed|iqr|rank) " "$log" | sed 's/^/    /'
        n=$((n+1))
      else
        echo "  !! FAILED (see $log)"; tail -6 "$log" | sed 's/^/     /'
        FAILED="$FAILED denom:${arm}/L${L}"
      fi
    done; done; done
  done
  if [ "$n" -eq 0 ]; then
    echo "  NO cells produced. Not a silent skip — please say so rather than"
    echo "  reporting this stage as done."
    FAILED="$FAILED stage2-empty"
  else
    echo "  [STAGE 2] $n cell(s)"; RAN=$((RAN+1))
  fi
fi

# =========================================================== STAGE 3  d_struct seeds
if want 3; then
  banner "STAGE 3  d_struct at three model seeds (CPU)"
  echo "  gates: $GATES"
  mkdir -p results_interplm_metric
  n=0
  for root in "$OUT_NATIVE" "$OUT_SHUF"; do
    [ -d "$root" ] || { echo "  [skip] no $root/"; continue; }
    for s in $SEEDS; do for arm in $(arms_for_seed "$s"); do for L in $DEPTHS; do
      d="$root/$arm/layer_$L"
      [ -f "$d/Z.npy" ] || { echo "  [skip] no Z.npy in $d"; continue; }
      for g in $GATES; do
        out="results_interplm_metric/${root##*/}_${arm}_L${L}_gate-${g}.csv"
        [ -f "$out" ] && { echo "  [done] $(basename "$out")"; n=$((n+1)); continue; }
        echo "=== [dstruct] $arm L$L gate=$g $(date +%H:%M:%S) ==="
        log="$LOGDIR/s3_dstruct_${root##*/}_${arm}_L${L}_${g}_${STAMP}.log"
        if $PY -u experiment_interplm_metric.py --layer-dir "$d" \
               --gate-mode "$g" --out "$out" > "$log" 2>&1; then
          tail -3 "$log" | sed 's/^/    /'; n=$((n+1))
        else
          echo "  !! FAILED (see $log)"; tail -6 "$log" | sed 's/^/     /'
          FAILED="$FAILED dstruct:${arm}/L${L}/${g}"
        fi
      done
    done; done; done
  done
  if [ "$n" -eq 0 ]; then
    echo "  NO cells produced. Not a silent skip."
    FAILED="$FAILED stage3-empty"
  else
    echo "  [STAGE 3] $n cell(s)"; RAN=$((RAN+1))
  fi
fi

# =========================================================== STAGE 4  untrained d_struct
if want 4; then
  banner "STAGE 4  d_struct untrained baseline (GPU forward pass, then CPU)"
  echo "  --randomize-model discards the trained weights and seeds off --sae-seed,"
  echo "  so the checkpoint is read for its architecture only. No pretraining."
  mkdir -p results_interplm_metric
  n=0
  for s in $SEEDS; do
    for arm in $(arms_for_seed 42); do
      CK="$CKPT_ROOT/$arm/model_final.pt"
      [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
      for L in $DEPTHS; do
        root="outputs_ctrl_randominit_s${s}"
        d="$root/$arm/layer_$L"
        if [ ! -f "$d/Z.npy" ]; then
          echo "=== [randinit embed] s$s $arm L$L $(date +%H:%M:%S) ==="
          log="$LOGDIR/s4_embed_s${s}_${arm}_L${L}_${STAMP}.log"
          if ! $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$arm" --layer "$L" \
                 --out-root "$root" --eval-set eval_set \
                 --randomize-model --sae-seed "$s" \
                 --k-sparse "$K_SPARSE" --expansion "$EXPANSION" > "$log" 2>&1; then
            echo "  !! FAILED (see $log)"; tail -6 "$log" | sed 's/^/     /'
            FAILED="$FAILED randinit-embed:s${s}/${arm}/L${L}"; continue
          fi
        fi
        for g in $GATES; do
          out="results_interplm_metric/randominit_s${s}_${arm}_L${L}_gate-${g}.csv"
          [ -f "$out" ] && { echo "  [done] $(basename "$out")"; n=$((n+1)); continue; }
          echo "=== [dstruct-untrained] s$s $arm L$L gate=$g $(date +%H:%M:%S) ==="
          log="$LOGDIR/s4_dstruct_s${s}_${arm}_L${L}_${g}_${STAMP}.log"
          if $PY -u experiment_interplm_metric.py --layer-dir "$d" \
                 --gate-mode "$g" --out "$out" > "$log" 2>&1; then
            tail -3 "$log" | sed 's/^/    /'; n=$((n+1))
          else
            echo "  !! FAILED (see $log)"; tail -6 "$log" | sed 's/^/     /'
            FAILED="$FAILED dstruct-untrained:s${s}/${arm}/L${L}/${g}"
          fi
        done
      done
    done
  done
  if [ "$n" -eq 0 ]; then
    echo "  NO cells produced. Not a silent skip."
    FAILED="$FAILED stage4-empty"
  else
    echo "  [STAGE 4] $n cell(s)"; RAN=$((RAN+1))
  fi
fi

# ------------------------------------------------------------------ package
banner "packaging"
PKG="tier1_results_$(date +%Y%m%d)"
rm -rf "$PKG"; mkdir -p "$PKG"
for root in "$OUT_NATIVE" "$OUT_SHUF"; do
  [ -d "$root" ] || continue
  find "$root" -name struct_seq_metrics_denominators.csv -print0 2>/dev/null |
    while IFS= read -r -d '' f; do mkdir -p "$PKG/$(dirname "$f")"; cp "$f" "$PKG/$f"; done
done
[ -d results_interplm_metric ] && cp -r results_interplm_metric "$PKG/" 2>/dev/null
[ -f results_rigor/top1_agreement.csv ] && { mkdir -p "$PKG/results_rigor"; cp results_rigor/top1_agreement.csv "$PKG/results_rigor/"; }
mkdir -p "$PKG/logs"; cp "$LOGDIR"/*_"$STAMP".log "$PKG/logs/" 2>/dev/null
git rev-parse HEAD > "$PKG/git_revision.txt" 2>/dev/null
git status --short > "$PKG/git_status.txt" 2>/dev/null
# sha256sum on Linux, shasum -a 256 on macOS; neither is guaranteed
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"
elif command -v shasum >/dev/null 2>&1; then SHA="shasum -a 256"
else SHA=""; fi
if [ -n "$SHA" ]; then
  ( cd "$PKG" && find . -type f ! -name SHA256SUMS.txt -exec $SHA {} + > SHA256SUMS.txt 2>/dev/null )
else
  echo "  note: no sha256sum/shasum on PATH — SHA256SUMS.txt not written"
fi

{
  echo "# Tier 1 batch"
  echo
  echo "- Date: $(date +%Y-%m-%d)"
  echo "- Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "- Seeds: $SEEDS | Depths: $DEPTHS | n_shuffles: $NSHUF | gates: $GATES"
  echo "- Denominators: $DENOMS"
  echo "- Stages run: $RAN"
  echo
  echo "## Mean struct_delta by denominator"
  echo
  echo "| condition | arm | layer | sd | fixed | iqr | rank | self-check |"
  echo "|---|---|---:|---:|---:|---:|---:|---|"
  for root in "$OUT_NATIVE" "$OUT_SHUF"; do
    [ -d "$root" ] || continue
    find "$root" -name struct_seq_metrics_denominators.csv 2>/dev/null | sort |
    while read -r f; do
      arm=$(echo "$f" | awk -F/ '{print $2}'); lay=$(echo "$f" | awk -F/ '{print $3}')
      $PY - "$f" "$root" "$arm" "${lay#layer_}" <<'PYEOF'
import csv, sys, statistics as st
path, root, arm, lay = sys.argv[1:5]
with open(path) as fh:
    rows = list(csv.DictReader(fh))
def m(c):
    k = f"struct_delta_{c}"
    if not rows or k not in rows[0]:
        return "-"
    return f"{st.mean(float(r[k]) for r in rows):+.5f}"
print(f"| {root} | {arm} | {lay} | {m('sd')} | {m('fixed')} | {m('iqr')} | {m('rank')} | see log |")
PYEOF
    done
  done
  echo
  echo "## d_struct cells"
  echo
  ls -1 results_interplm_metric 2>/dev/null | sed 's/^/- /' || echo "(none)"
  [ -n "$FAILED" ] && { echo; echo "## FAILED"; echo; echo "$FAILED"; }
} > "$PKG/RESULT_SUMMARY.md"

tar czf "${PKG}.tgz" "$PKG"
echo "  wrote ${PKG}.tgz  ($(du -h "${PKG}.tgz" | cut -f1))"

# ------------------------------------------------------- delegated / excluded
banner "already scripted — run these separately, this script does not duplicate them"
cat <<'DELEG'
  100-permutation null
      NSHUF_HI=100 ONLY=2 bash run_checks.sh
      run_checks.sh stage 2 already drives this; only the value changes.

  Fold-disjoint refit of the corpus control
      FOLDDISJ_APPLY=1 ONLY=3 bash run_checks.sh     # native arm
      STAGE=3 bash run_gaps.sh                       # shuffled arm (branch runs/rescore-batch)
      The shuffled half ran in the 2026-08-28 batch. The native CSVs exist on
      this box and were not in that archive — send
      outputs_ctrl_folddisj/*/*/struct_seq_metrics.csv.

  Probes at block 18
      ONLY=2 bash RUN_DEPTH_GRID.sh

  Pairwise contact probes, SAE and raw (tests the co-activation assumption)
      STAGE=11 bash RUN_MUSTRUNS.sh
      ~1 h CPU. experiment_pairwise_probe.py already exists and is already wired
      into RUN_MUSTRUNS stage 11, with shuffled-label and separation-matched
      controls and a positive control.

  Evaluation-distribution control
      Already run, 2026-08-28, run_gaps.sh stage 1 -> outputs_ctrl_evaldist/.

Not queued, on cost:
  Unify frozen-dictionary depths to 11/14/18   days (retrains shared dictionaries)
  concept-F1 on the Swiss-Prot concept set     ~68 h  (interplm_attack/)
  Block-shuffle destruction, scaling seeds     needs pretraining
DELEG

echo
if [ -n "$FAILED" ]; then
  echo "FAILURES:$FAILED"
  echo "Send the archive anyway — partial cells are usable."
  exit 1
fi
if [ "$RAN" -eq 0 ]; then
  echo "NOTHING RAN. That is a failure, not a success — check ONLY= and the preflight."
  exit 1
fi
echo "All stages OK ($RAN run). Send ${PKG}.tgz."
