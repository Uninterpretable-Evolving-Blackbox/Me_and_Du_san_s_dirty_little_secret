#!/usr/bin/env bash
# run_checks.sh — the verification queue. RUN THIS WHEN JOB 3 FINISHES, NOT JOB 4.
#
#   bash run_checks.sh
#
# Job 4 (RUN_500TPP_SEEDS.md) is still on hold and still costs ~200 GPU-hours. These checks
# cost about three, and two of the three need no GPU at all. See RUN_CHECKS.md for why.
#
# Same discipline as run_all_pending.sh: every stage is idempotent and skip-if-done, a
# failing stage never stops the queue, and failures are collected and printed at the end.
#
#   1  grid audit — is the grid usable at all?            seconds   reads files, no compute
#   2  permutation null at 25 on the headline cells      ~1-2 h    CPU only
#   3  fold-disjoint split: price it, then fix it        ~5 min,   report is free;
#                                                        then 1-3 h  the re-fit needs the GPU
#   4  concept scores against their prevalence floor     ~30 min   CPU only
#   5  global level: remote-homology probe               ~1 h      CPU only
#
# Run one stage:    ONLY=2 bash run_checks.sh
# Skip a stage:     SKIP="3" bash run_checks.sh
# CPU-only subset:  SKIP="3" bash run_checks.sh   (1,2,4,5 need no GPU at all)
#
# Stage 3 only REPORTS by default. It rebuilds dictionaries only with:
#     FOLDDISJ_APPLY=1 bash run_checks.sh
# because if the current split turns out to be near fold-disjoint already, the honest fix is
# one sentence in the paper rather than hours of GPU time — and stage 3 tells you which.
set -u
cd "$(dirname "$0")"

_REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
_DIRTY=$(git diff --quiet 2>/dev/null && echo "" || echo " +local-changes")
echo "########## repo revision: ${_REV}${_DIRTY} | started $(date) ##########"

PY="${PY:-python}"
OUT="${OUT:-outputs_ctrl}"
BASE="${BASE:-$HOME/interplm_stress}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
LOGDIR="logs_checks"; mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FAILED=""

SEEDS="${SEEDS:-42 43 44}"
DEPTHS_HEADLINE="${DEPTHS_HEADLINE:-11 14 18}"
NSHUF_HI="${NSHUF_HI:-25}"          # the enlarged permutation null; the paper uses 5
NSHUF_OUT="${NSHUF_OUT:-results_nshuffle_headline}"
FOLDDISJ_APPLY="${FOLDDISJ_APPLY:-0}"
RAW_ROOT="${RAW_ROOT:-outputs_raw_real}"          # STAGE=10 writes <arm>/layer_<L>/X.npy
RANDOMINIT_ROOT="${RANDOMINIT_ROOT:-outputs_ctrl_randominit}"   # STAGE=15
K_SPARSE="${K_SPARSE:-256}"
EXPANSION="${EXPANSION:-8}"

# ------------------------------------------------------------------ preflight
# Stages 2, 4 and 5 all read Z.npy, and Z is pruned after the main controlled run. Say so
# ONCE, up front, rather than letting it surface as eighteen per-cell skips in stage two.
_n_z=0; _n_want=0
for _s in $SEEDS; do
  for _a in "ckpt_mlm_s${_s}_token" "ckpt_clm_s${_s}"; do
    for _L in $DEPTHS_HEADLINE; do
      _n_want=$((_n_want+1))
      [ -f "$OUT/$_a/layer_$_L/Z.npy" ] && _n_z=$((_n_z+1))
    done
  done
done
echo "preflight: $_n_z / $_n_want headline cells have Z.npy in $OUT/"
if [ "$_n_z" -eq 0 ]; then
  echo
  echo "  !! NO Z.npy ANYWHERE. Stages 2, 4 and 5 all read it and will all fail."
  echo "     Z is pruned after the main controlled run, so this is expected rather than broken."
  echo "     Rebuild first (~2 h), then re-run this script:"
  echo "         STAGE=1 bash RUN_MUSTRUNS.sh"
  echo "     Stage 3 does not need Z and will still work:  ONLY=3 bash run_checks.sh"
  echo
elif [ "$_n_z" -lt "$_n_want" ]; then
  echo "  NOTE: $((_n_want-_n_z)) headline cell(s) have no Z.npy and will be skipped by 2/4/5."
fi
[ -d "$RANDOMINIT_ROOT" ] || echo "preflight: no $RANDOMINIT_ROOT/ — stage 4 loses its random-init contrast (STAGE=15)"
[ -d "$RAW_ROOT" ] || echo "preflight: no $RAW_ROOT/ — stage 5 loses its raw-vs-SAE contrast (STAGE=10)"
echo

want () {
  local n="$1"
  [ -n "$ONLY" ] && [ "$ONLY" != "$n" ] && return 1
  case " $SKIP " in *" $n "*) return 1 ;; esac
  return 0
}

run () {    # run <n> <name> <command...>
  local n="$1" name="$2"; shift 2
  want "$n" || { echo "  [skip stage $n] $name"; return 0; }
  local log="$LOGDIR/check${n}_${STAMP}.log"
  echo
  echo "===== CHECK $n: $name — started $(date) -> $log ====="
  if "$@" > "$log" 2>&1; then
    echo "===== CHECK $n OK $(date) ====="
    tail -20 "$log" | sed 's/^/    /'
  else
    echo "===== CHECK $n FAILED (rc=$?) $(date) — CONTINUING ====="
    FAILED="$FAILED $n"
    tail -25 "$log" | sed 's/^/    /'
  fi
}

# ------------------------------------------------------------------ 1: grid audit
# Reads $BASE/results/{sae_quality.txt,concept_f1.txt}. Answers two things before anyone
# quotes an F1: how many cells exist per arm, and whether the two arms hold comparable
# dictionaries. A partial grid is masked-heavy by construction, because each layer instance
# trains masked seeds before causal ones.
run 1 "grid audit — completeness and arm comparability" \
  $PY -u check_grid.py --base "$BASE"

# ------------------------------------------------------------------ 2: permutation null
# A larger permutation null is flagged essential in the supervisor feedback. results_nshuffle_sensitivity/ already
# has 5-vs-25 for the CONTACT-DEFINITION SWEEP, but every headline L_struct number is still
# at 5. This runs the 18 headline cells at 25 and diffs them.
#
# cpu_stage.py writes struct_seq_metrics.csv INTO the directory it is given, so running it
# on the real layer dir would overwrite the 5-shuffle result the paper reports. Every cell
# below therefore runs in a scratch directory of SYMLINKS to the real inputs: no copying of
# 1.5 GB Z files, and the originals are physically unreachable for writing.
stage_nshuf () {
  local rc=0 n_done=0 n_skip=0
  for s in $SEEDS; do
    for arm in "ckpt_mlm_s${s}_token" "ckpt_clm_s${s}"; do
      for L in $DEPTHS_HEADLINE; do
        local LD="$OUT/$arm/layer_$L"
        local SC="$NSHUF_OUT/nshuf${NSHUF_HI}/$arm/layer_$L"
        if [ ! -f "$LD/Z.npy" ]; then
          echo "  [skip] no $LD/Z.npy — regenerate with eval_ctrl_plm.py or STAGE=1 RUN_MUSTRUNS.sh"
          n_skip=$((n_skip+1)); continue
        fi
        if [ -f "$SC/struct_seq_metrics.csv" ]; then
          echo "  [done] $arm L$L"; n_done=$((n_done+1)); continue
        fi
        mkdir -p "$SC"
        for f in Z.npy uids.json lengths.npy sequences.json D.npy META.json; do
          [ -e "$LD/$f" ] && ln -sfn "$(readlink -f "$LD/$f")" "$SC/$f"
        done
        echo "=== [nshuf $NSHUF_HI] $arm L$L $(date +%H:%M:%S) ==="
        if $PY -u cpu_stage.py --layer-dir "$SC" --model-type residue \
                --n-shuffles "$NSHUF_HI"; then
          n_done=$((n_done+1))
        else
          echo "  [fail] $arm L$L"; rc=1
        fi
      done
    done
  done
  echo
  echo "  cells at n-shuffles=$NSHUF_HI: $n_done done, $n_skip skipped for a missing Z"
  [ "$n_skip" -gt 0 ] && echo "  NOTE: skipped cells are NOT in the comparison below."
  if [ "$n_done" -eq 0 ]; then
    echo
    echo "  PRODUCED NOTHING. Every cell was skipped, so there is no comparison to read."
    echo "  This is a failure, not a pass — do not send an empty results directory."
    rc=1
  fi

  # 5 vs 25, per cell. A stable mean is the result we want; a moving one means the paper's
  # numbers are sensitive to the size of the null and the null has to grow.
  OUT="$OUT" NSHUF_OUT="$NSHUF_OUT" NSHUF_HI="$NSHUF_HI" \
  SEEDS="$SEEDS" DEPTHS_HEADLINE="$DEPTHS_HEADLINE" $PY - <<'PYEOF'
import os, csv, statistics as st
OUT=os.environ["OUT"]; NO=os.environ["NSHUF_OUT"]; HI=os.environ["NSHUF_HI"]
seeds=os.environ["SEEDS"].split(); depths=os.environ["DEPTHS_HEADLINE"].split()
def mean_delta(p):
    if not os.path.exists(p): return None
    v=[float(r["struct_delta"]) for r in csv.DictReader(open(p))
       if r.get("struct_delta") not in (None,"","nan")]
    return st.mean(v) if v else None
print()
print("  %-26s %10s %10s %10s" % ("cell","n=5","n=%s"%HI,"abs diff"))
diffs=[]
for s in seeds:
    for arm in (f"ckpt_mlm_s{s}_token", f"ckpt_clm_s{s}"):
        for L in depths:
            a=mean_delta(f"{OUT}/{arm}/layer_{L}/struct_seq_metrics.csv")
            b=mean_delta(f"{NO}/nshuf{HI}/{arm}/layer_{L}/struct_seq_metrics.csv")
            if a is None or b is None: continue
            diffs.append(abs(a-b))
            print("  %-26s %10.5f %10.5f %10.5f" % (f"{arm} L{L}", a, b, abs(a-b)))
if diffs:
    print()
    print("  max |diff| = %.5f   median |diff| = %.5f  over %d cells"
          % (max(diffs), st.median(diffs), len(diffs)))
    print("  Read it as: a small max means the 5-permutation null was already large enough")
    print("  and the published numbers stand. A large one means the null has to grow and")
    print("  every L_struct number moves with it.")
else:
    print("  no comparable cells — nothing to diff.")
PYEOF
  return $rc
}
run 2 "permutation null at $NSHUF_HI on the 18 headline cells" stage_nshuf

# ------------------------------------------------------------------ 3: fold-disjoint split
# The one methodological asymmetry a reviewer will find: concept-F1 is fold-disjoint, the
# 1,350/150 split that fits the dictionaries and probes is uniform random. Price it first --
# if the current split already has near-zero fold leakage, the fix is a sentence, not a re-fit.
stage_folddisj () {
  local rc=0
  echo "=== pricing the current split ==="
  $PY -u make_folddisjoint_eval_set.py --eval-set eval_set --out eval_set_folddisj || rc=1
  if [ "$FOLDDISJ_APPLY" != "1" ]; then
    echo
    echo "  REPORT ONLY. eval_set_folddisj/ is written but no dictionary was re-fitted."
    echo "  Send the numbers above first. To actually re-fit:  FOLDDISJ_APPLY=1 bash run_checks.sh"
    return $rc
  fi
  [ -d eval_set_folddisj ] || { echo "  no eval_set_folddisj/ — cannot apply"; return 1; }
  for s in $SEEDS; do
    for arm in "ckpt_mlm_s${s}_token" "ckpt_clm_s${s}"; do
      local CK="$HOME/own_sae_data/uniref50_pilot/$arm/model_final.pt"
      [ -f "$CK" ] || { echo "  [skip] no $CK"; continue; }
      for L in $DEPTHS_HEADLINE; do
        local DEST="outputs_ctrl_folddisj/$arm/layer_$L"
        if [ -f "$DEST/struct_seq_metrics.csv" ]; then echo "  [done] $arm L$L"; continue; fi
        echo "=== [folddisj] $arm L$L $(date +%H:%M:%S) ==="
        $PY -u eval_ctrl_plm.py --ckpt "$CK" --name "$arm" --layer "$L" \
            --out-root outputs_ctrl_folddisj --eval-set eval_set_folddisj \
            --k-sparse "$K_SPARSE" --expansion "$EXPANSION" || { rc=1; continue; }
        $PY -u cpu_stage.py --layer-dir "$DEST" --model-type residue --n-shuffles 5 || rc=1
      done
    done
  done
  return $rc
}
run 3 "fold-disjoint SAE/probe split (report; re-fit only with FOLDDISJ_APPLY=1)" stage_folddisj

# ------------------------------------------------------------------ 4: trivial baseline
# How much of a concept score is just label prevalence? A label with prevalence p is scored
# 2p/(p+1) by the classifier that marks every residue positive, and helix/burial are prevalent
# enough that this floor is most of the reportable range. Every quoted concept number needs
# its floor next to it.
#
# Measured: 99 s for a 3,840-feature dictionary on a laptop, so ~1 min per controlled cell.
stage_trivial () {
  local rc=0 n_cells=0
  for s in $SEEDS; do
    for arm in "ckpt_mlm_s${s}_token" "ckpt_clm_s${s}"; do
      for L in $DEPTHS_HEADLINE; do
        local LD="$OUT/$arm/layer_$L"
        local DEST="results_trivial_baseline/${arm}_L${L}"
        [ -f "$LD/Z.npy" ] || { echo "  [skip] no $LD/Z.npy"; continue; }
        [ -f "$DEST/trivial_baseline.csv" ] && { echo "  [done] $arm L$L"; n_cells=$((n_cells+1)); continue; }
        echo "=== [trivial] $arm L$L $(date +%H:%M:%S) ==="
        $PY -u experiment_trivial_baseline.py --layer-dir "$LD" --out "$DEST" || rc=1
        n_cells=$((n_cells+1))
      done
    done
  done
  # the untrained arm is the informative contrast: it should sit AT the floor.
  # Random-init cells live under their own root, per arm (RUN_MUSTRUNS.sh:677).
  local n_ri=0
  for s in $SEEDS; do
    for arm in "ckpt_mlm_s${s}_token" "ckpt_clm_s${s}"; do
      for L in $DEPTHS_HEADLINE; do
        local LD="$RANDOMINIT_ROOT/$arm/layer_$L"
        [ -f "$LD/Z.npy" ] || continue
        local DEST="results_trivial_baseline/randominit_${arm}_L${L}"
        [ -f "$DEST/trivial_baseline.csv" ] && { n_ri=$((n_ri+1)); continue; }
        echo "=== [trivial/random-init] $arm L$L $(date +%H:%M:%S) ==="
        $PY -u experiment_trivial_baseline.py --layer-dir "$LD" --out "$DEST" || rc=1
        n_ri=$((n_ri+1))
      done
    done
  done
  if [ "$n_cells" -eq 0 ]; then
    echo
    echo "  PRODUCED NOTHING — no trained cell had a Z.npy. This is a failure, not a pass."
    rc=1
  fi
  if [ "$n_ri" -eq 0 ]; then
    echo
    echo "  NO random-init cells found under $RANDOMINIT_ROOT/ — run STAGE=15 bash RUN_MUSTRUNS.sh."
    echo "  Without them the trained margins have nothing to be read against, which is the"
    echo "  whole point of this stage. Not a silent skip: please say if they are gone."
  fi
  return $rc
}
run 4 "trivial baseline — concept scores against their prevalence floor" stage_trivial

# ------------------------------------------------------------------ 5: global-level probe
# The readout suite was meant to separate position-wise, pairwise and GLOBAL levels.
# Position-wise (linear probes) and pairwise (contact probes) exist; global never did.
# This is remote-homology detection between whole domains: same fold / different superfamily
# against different fold, split so no superfamily is on both sides.
#
# Measured: 268 s for a 3,840-feature dictionary at --n-pairs 5000 on a laptop, so ~3 min per
# controlled cell. Read the shuffled-label control FIRST -- it must sit at ~0.5.
stage_global () {
  local rc=0 n_cells=0
  for s in $SEEDS; do
    for arm in "ckpt_mlm_s${s}_token" "ckpt_clm_s${s}"; do
      for L in $DEPTHS_HEADLINE; do
        local LD="$OUT/$arm/layer_$L"
        local DEST="results_global_probe/${arm}_L${L}"
        [ -f "$LD/Z.npy" ] || { echo "  [skip] no $LD/Z.npy"; continue; }
        [ -f "$DEST/global_probe.csv" ] && { echo "  [done] $arm L$L"; n_cells=$((n_cells+1)); continue; }
        echo "=== [global] $arm L$L $(date +%H:%M:%S) ==="
        $PY -u experiment_global_probe.py --layer-dir "$LD" --out "$DEST" || rc=1
        n_cells=$((n_cells+1))
        # raw contrast. Same location the pairwise probe uses (RUN_MUSTRUNS.sh:510), so
        # STAGE=10 having been run is the only precondition.
        local RAW="$RAW_ROOT/$arm/layer_$L/X.npy"
        if [ -f "$RAW" ]; then
          [ -f "${DEST}_raw/global_probe.csv" ] || \
            $PY -u experiment_global_probe.py --layer-dir "$LD" --mode raw \
                --raw-npy "$RAW" --out "${DEST}_raw" || rc=1
        else
          echo "  [no raw] $RAW missing — run STAGE=10 bash RUN_MUSTRUNS.sh."
          echo "           The raw-vs-SAE contrast is the half that answers whether the"
          echo "           dictionary earns its place at this level."
        fi
      done
    done
  done
  if [ "$n_cells" -eq 0 ]; then
    echo
    echo "  PRODUCED NOTHING — no cell had a Z.npy. This is a failure, not a pass."
    rc=1
  fi
  return $rc
}
run 5 "global level — remote-homology probe between domains" stage_global

# ------------------------------------------------------------------ done
echo
echo "########## finished $(date) ##########"
if [ -n "$FAILED" ]; then
  echo "FAILED stages:$FAILED — logs in $LOGDIR/"
else
  echo "All stages OK."
fi
echo
echo "Send back:"
echo "  $LOGDIR/                     (carries the git revision of the code that ran)"
echo "  $NSHUF_OUT/                  (stage 2)"
echo "  eval_set_folddisj/META.json  (stage 3)"
echo "  results_trivial_baseline/    (stage 4)"
echo "  results_global_probe/        (stage 5)"
echo "  outputs_ctrl_folddisj/*/*/struct_seq_metrics.csv   (stage 3, only with FOLDDISJ_APPLY=1)"
echo "  and \$BASE/results/sae_quality.txt ALONGSIDE concept_f1.txt — never the F1 alone."
