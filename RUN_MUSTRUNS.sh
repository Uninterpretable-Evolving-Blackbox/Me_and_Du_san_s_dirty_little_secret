#!/usr/bin/env bash
# ============================================================================
# RUN_MUSTRUNS.sh — the outstanding experiments, in value order.
#
#   bash RUN_MUSTRUNS.sh            # stages 0-4 and 6-9
#   STAGE=6 bash RUN_MUSTRUNS.sh    # the composition test -- fast, do this first
#   STAGE=1 bash RUN_MUSTRUNS.sh    # one stage only
#   STAGE=5 bash RUN_MUSTRUNS.sh    # the long one: two more shuffled training runs
#
# Every stage is idempotent: it skips if its output CSV already exists.
# Nothing here deletes anything. Logs go to logs_mustruns/.
#
# TRAPS THIS SCRIPT HANDLES FOR YOU, because they have bitten before:
#   * --n-shuffles 5, never the default 3. Three computes a DIFFERENT metric
#     than every other number in the project.
#   * KEEP_Z=1 wherever Z.npy is needed downstream. prune_z has silently
#     removed the input of a later stage before.
#   * Stage exit codes are checked; a failure does not print OK.
# ============================================================================
set -uo pipefail

STAGE="${STAGE:-0}"
LOGS=logs_mustruns; mkdir -p "$LOGS"
REV="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
FAILED=()
say(){ printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok(){  printf '   \033[32mOK\033[0m  %s\n' "$*"; }
bad(){ printf '   \033[31m!! %s\033[0m\n' "$*"; FAILED+=("$*"); }
run(){ # run <logname> <cmd...>
  local n="$1"; shift
  echo "   -> $*" | tee "$LOGS/$n.log"
  echo "   [rev $REV] $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOGS/$n.log"
  if "$@" >> "$LOGS/$n.log" 2>&1; then ok "$n"; return 0
  else bad "$n FAILED — see $LOGS/$n.log"; return 1; fi
}

CTRL_REAL="${CTRL_REAL:-outputs_ctrl}"
CTRL_SHUF="${CTRL_SHUF:-outputs_ctrl_shuf}"
MLM="${MLM:-ckpt_mlm_s42_token}"
CLM="${CLM:-ckpt_clm_s42}"
LAYERS="${LAYERS:-11,14,18}"

# There is no `python` on PATH in a non-interactive shell on the RTX box, so every
# stage died instantly with "command not found" the first time this ran. Resolve an
# interpreter explicitly, matching the other scripts in this repo.
PY="${PY:-$PWD/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "!! no python interpreter found; set PY=/path/to/python"; exit 2; }
echo "   interpreter: $PY"


# ---------------------------------------------------------------------------
# STAGE 0 — rebuild Z.npy where it is missing.  RUN THIS FIRST.
#
# WHY: prune_z removes Z.npy after each pipeline run (6-7 GB/cell), so the
# controlled and shuffled trees almost certainly have none. Stage 1 needs it and
# will otherwise print "[skip] Z.npy missing" for every layer and produce nothing.
#
# SAFETY (same argument as fix_pred_bootstrap.sh, which did this successfully):
# eval_ctrl_plm.py seeds torch+numpy from --sae-seed immediately before SAE
# training and reads the val split from META.json, so regeneration reproduces the
# SAME SAE that produced the existing struct_seq_metrics.csv - last run this came
# back identical to the last digit. It does NOT write struct_seq_metrics.csv, so
# delivered results are untouched. META.json IS rewritten deterministically; we
# back it up first anyway.
#
# COST: this retrains an SAE per cell (~9 min). 12 cells ~= 2 h. It is the real
# cost of stage 1, not the sweep itself.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 1 ]; then
  say "STAGE 0 — rebuild Z.npy where missing (needed by stage 1)"
  need=0
  for spec in "$CTRL_REAL:uniref50_pilot" "$CTRL_SHUF:uniref50_pilot_shuf"; do
    root="${spec%%:*}"; data="${spec##*:}"
    for arm in "$MLM" "$CLM"; do
      for L in ${LAYERS//,/ }; do
        d="$root/$arm/layer_$L"
        [ -f "$d/Z.npy" ] && continue
        need=$((need+1))
        ck="$HOME/own_sae_data/$data/$arm/model_final.pt"
        [ -f "$ck" ] || { bad "no checkpoint for $arm in $data - cannot rebuild $d"; continue; }
        [ -f "$d/META.json" ] && cp "$d/META.json" "$d/META.json.bak.$(date -u +%Y%m%dT%H%M%SZ)"
        run "s0_${root##*/}_${arm}_L${L}" "$PY" eval_ctrl_plm.py \
          --ckpt "$ck" --name "$arm" --layer "$L" --out-root "$root" \
          --eval-set eval_set --sae-seed 42 --expansion 8 --k-sparse 256
      done
    done
  done
  [ "$need" = 0 ] && ok "Z.npy already present everywhere stage 1 needs it"
  echo "   (rebuilt $need cell(s); struct_seq_metrics.csv was not touched)"
fi

# ---------------------------------------------------------------------------
# STAGE 1 — the metric at Simon & Zou's own settings.  HIGHEST VALUE.
#
# WHY: our L_struct adds a minimum sequence separation (|i-j| >= 12) that
# InterPLM does not use. Section 5.3 shows that floor carries 81-93% of the
# effect. So we do not currently know whether the shuffled-input failure
# belongs to the published recipe or specifically to our modification.
#
# This stage answers that, and the two answers imply DIFFERENT PAPERS:
#   ratio ~46x at 6A/gap1  -> the published recipe fails too; claim broadens
#   ratio ~2x  at 6A/gap1  -> the failure tracks OUR separation floor; claim
#                             narrows, and the finding becomes "making the
#                             metric stricter is what made it invalid"
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 1 ]; then
  say "STAGE 1 — InterPLM settings (6 A, no separation floor) vs our default"
  for root in "$CTRL_REAL" "$CTRL_SHUF"; do
    tag=$(basename "$root")
    outdir="results_interplm_def_${tag}"
    if [ -f "$outdir/contact_def_sweep.csv" ]; then ok "$tag cached"; continue; fi
    [ -d "$root/$MLM" ] || { bad "$root/$MLM missing — skipping"; continue; }
    export KEEP_Z=1   # the sweep reads Z.npy; prune_z must not remove it
    run "s1_${tag}" "$PY" experiment_contact_def_sweep.py \
      --root "$root" --model-a "$MLM" --model-b "$CLM" \
      --layers "$LAYERS" --n-shuffles 5 \
      --cutoffs 6 --gaps 1,2,12 \
      --out "$outdir"
  done
  echo "   compare: results_interplm_def_${CTRL_SHUF##*/}/contact_def_sweep.csv"
  echo "            / results_interplm_def_${CTRL_REAL##*/}/contact_def_sweep.csv"
  echo "   the shuffled/real ratio at (6 A, gap 1) is the number that decides it."
fi

# ---------------------------------------------------------------------------
# STAGE 2 — amino-acid selectivity on the SHUFFLED checkpoints.
# WHY: shuffling permutes rather than resamples, so composition is the one
# thing it preserves exactly. "The inflation is composition-driven" is the
# live alternative explanation and has never been tested on the models it
# concerns — only on the published pair.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 2 ]; then
  say "STAGE 2 — AA selectivity on the shuffled checkpoints"
  if [ -f results_rigor/aa_selectivity_shuf.csv ]; then ok "cached"; else
    run s2_aa_selectivity "$PY" experiment_aa_selectivity.py \
      --root "$CTRL_SHUF" --cells "${MLM}:14,${CLM}:14" \
      --out results_rigor/aa_selectivity_shuf.csv
  fi
fi

# ---------------------------------------------------------------------------
# STAGE 3 — regenerate val_EV at every depth and seed.
# WHY: val_EV was logged for the real models at only THREE depths, so the
# reconstruction-gain correlation rests on n=6 (3 depths x 2 arms, seed 42).
# This is NOT a re-analysis: the metadata does not exist for the other cells,
# so the statistic has to be recomputed from activations.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 3 ]; then
  say "STAGE 3 — val_EV at all 9 depths, 3 seeds, both arms"
  for s in 42 43 44; do
    for arm in "mlm_s${s}_token" "clm_s${s}"; do
      ck="$HOME/own_sae_data/uniref50_pilot/ckpt_${arm}/model_final.pt"
      out="results_rank_ev/valev_${arm}.json"
      [ -f "$out" ] && { ok "$arm cached"; continue; }
      [ -f "$ck" ]  || { bad "checkpoint missing: $ck"; continue; }
      run "s3_${arm}" "$PY" measure_rank_ev.py \
        --ckpt "$ck" --name "$arm" --eval-set eval_set --out "$out"
    done
  done
fi

# ---------------------------------------------------------------------------
# STAGE 4 — the MLP probe. Code already exists (commit c169023).
# WHY: a FALSIFICATION test, not a robustness check. If the depth reversal is
# a fact about linear separability rather than information content, the probe
# half of the paper loses its claim. Better to know.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 4 ]; then
  say "STAGE 4 — nonlinear (MLP) probe"
  if [ -f results_ctrl_saefree_mlp/saefree_by_arm.csv ]; then ok "cached"; else
    run s4_mlp_probe "$PY" eval_ctrl_saefree.py --mlp \
      --out results_ctrl_saefree_mlp --skip-contacts \
      --arms ckpt_mlm_s42_token,ckpt_clm_s42,ckpt_mlm_s43_token,ckpt_clm_s43,ckpt_mlm_s44_token,ckpt_clm_s44
  fi
fi

# ---------------------------------------------------------------------------
# STAGE 5 — two more shuffled training runs (seeds 43, 44). LONG.
# WHY: the shuffled control and the L_seq positive control are both seed 42
# only, because only one shuffled pair was ever trained.
# Not run by default.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 5 ]; then
  say "STAGE 5 — shuffled corpus + training, seeds 43 and 44"
  SHUF_DATA="$HOME/own_sae_data/uniref50_pilot_shuf"
  if [ ! -d "$SHUF_DATA" ]; then
    run s5_prep "$PY" prep_controlled_corpus.py --shuffle-residues \
      --out-dir "$SHUF_DATA"
  else ok "shuffled corpus present"; fi
  for s in 43 44; do
    for obj in mlm clm; do
      out="$HOME/own_sae_data/uniref50_pilot_shuf/ckpt_${obj}_s${s}"
      [ -d "$out" ] && { ok "${obj}_s${s} cached"; continue; }
      run "s5_${obj}_s${s}" "$PY" train_ctrl_plm.py \
        --data-dir "$SHUF_DATA" --objective "$obj" --seed "$s" \
        --data-order-seed 1234 --out-dir "$out"
    done
  done
fi


# ---------------------------------------------------------------------------
# STAGE 6 — does L_struct reward composition alone?  RUN THIS ONE FIRST.
#
# WHY: section 5.4 rejects "composition clustering" using an amino-acid
# SELECTIVITY measure -- one minus normalised entropy over the 20 types. That
# measure is structurally blind to a CLASS detector: a feature firing on all
# hydrophobic residues spreads over ~8 types and scores as maximally UNselective.
#
# The permutation null shuffles residue positions against a FIXED structure, so
# any property with spatial autocorrelation in the fold produces excess
# structural co-activation without encoding structure. Hydrophobicity is the
# obvious case: hydrophobic residues are disproportionately buried, buried
# residues have high contact degree, and their long-range partners are
# disproportionately hydrophobic too.
#
# WHAT IT DOES: builds synthetic "features" that are pure indicator functions of
# residue identity -- no model, no SAE, no training -- and pushes them through
# the IDENTICAL metric path (same graphs, same permutation null, same
# struct_delta = observed - shuffled).
#
# RESULT ON THE LAPTOP (33.2M pair, layer 6, 8 A / gap 12): a pure cysteine
# indicator scores +0.4221. The best of 3,840 REAL learned features in the same
# cell scores +0.3925. An indicator beats 100% of learned features. V/I/L clear
# the 99th percentile. Charged residues score negative. That is the burial
# pattern, and it means the composition account is NOT refuted.
#
# ONE RUN, at topk 0.60. A BINARY feature whose occupancy exceeds topk_frac is
# silently zeroed -- percentile(acts, 100*(1-topk)) is exactly 1.0, `acts > thresh`
# selects nothing, n_active = 0, d is forced to 0.0. class:small has occupancy
# 0.4929, so topk must sit above that; 0.60 leaves a 0.107 margin. The script now
# REFUSES to emit a zero rather than let it read as a null result.
#
# For a binary feature the score is INVARIANT to topk_frac provided topk > occupancy
# (verified: max |tk0.60 - tk0.50| = 0.0000000000 across all 28 features, and
# |tk0.60 - tk0.10| = 0.0000000000 across the 20 single types). So one run above the
# largest occupancy is sufficient and every row IS comparable to the paper's numbers.
#
# This degeneracy needs values TIED at the threshold, which only binary indicators
# produce. Real continuous SAE features are NOT affected -- checked in the 33.2M
# layer-6 cell: 824 of 3,840 real features have occupancy above 0.10 and NOT ONE has
# struct_delta == 0. See the retraction in RUN_SYNTHETIC.md.

if [ "$STAGE" = 0 ] || [ "$STAGE" = 6 ]; then
  say "STAGE 6 — synthetic composition indicators (no model)"
  for arm in "$MLM" "$CLM"; do
    for L in ${LAYERS//,/ }; do
      d="$CTRL_REAL/$arm/layer_$L"
      # NB this stage does NOT need Z.npy -- it builds its own features from the
      # sequences. Only the optional --compare-real rescoring reads Z.npy.
      [ -f "$d/sequences.json" ] && [ -f "$d/lengths.npy" ] || {
        bad "no sequences.json/lengths.npy in $d"; continue; }
      out="results_synthetic_composition/${arm}_L${L}.csv"
      [ -f "$out" ] && { ok "$(basename $out) cached"; continue; }
      mkdir -p results_synthetic_composition
      EXTRA=""
      [ -f "$d/Z.npy" ] && EXTRA="--compare-real"
      run "s6_${arm}_L${L}" "$PY" experiment_synthetic_composition.py \
        --layer-dir "$d" --topk-frac 0.60 --n-shuffles 5 --out "$out" $EXTRA
    done
  done
  echo "   the number to look at: best synthetic struct_delta vs the p99 and max"
  echo "   of the real features printed beneath it, per cell."
fi

# ---------------------------------------------------------------------------
# STAGE 7 — dictionary-seed variation.  The missing variance component.
#
# WHY: every number in the paper sits at SAE seed 42. The reported spread
# (SD 0.0234 across three seeds) is MODEL-seed variance only, and the paper's
# seed-level intervals are built on it. But the dictionary is instrument, not
# model, and Section 5.2 finds L_struct tracks dictionary capacity at rho +0.94.
# If dictionary-seed variance is comparable, those intervals are understated and
# the omnibus p = 0.0028 is optimistic. Nobody has ever measured it.
#
# DESIGN: model seed fixed at 42, capacity fixed at the paper's k=256/expansion=8,
# ONLY the SAE init seed moves. That is the correct pairing -- vary one seed axis
# with the other pinned, or the variance cannot be attributed.
#
# COST: 3 layers x 2 arms x 2 seeds = 12 SAE fits, ~9 min each, ~2 h. GPU.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 7 ]; then
  say "STAGE 7 — dictionary-seed variation (model seed fixed at 42)"
  for dseed in 43 44; do
    for arm in "$MLM" "$CLM"; do
      for L in ${LAYERS//,/ }; do
        root="outputs_ctrl_dseed${dseed}"
        [ -f "$root/$arm/layer_$L/struct_seq_metrics.csv" ] && { ok "dseed$dseed $arm L$L cached"; continue; }
        ck="$HOME/own_sae_data/uniref50_pilot/$arm/model_final.pt"
        [ -f "$ck" ] || { bad "no checkpoint for $arm - cannot run dseed$dseed"; continue; }
        run "s7_dseed${dseed}_${arm}_L${L}" "$PY" eval_ctrl_plm.py \
          --ckpt "$ck" --name "$arm" --layer "$L" --out-root "$root" \
          --eval-set eval_set --sae-seed "$dseed" --expansion 8 --k-sparse 256
      done
    done
  done
  echo "   compare mean struct_delta across outputs_ctrl{,_dseed43,_dseed44} per cell."
  echo "   the number that matters: SD across dictionary seeds vs SD across model seeds (0.0234)."
fi

# ---------------------------------------------------------------------------
# STAGE 8 — extend the capacity sweep until the two arms overlap in val_EV.
#
# WHY: Section 4.3 cannot say how much of the shuffled-model rise is the
# autoencoder, because the estimate ranges from 4% to 89% depending purely on
# which slope is fitted. The reason it is under-determined is that the two arms'
# capacity configurations are DISJOINT in val_EV -- masked 0.90-0.98, causal
# 0.26-0.84 -- so no common dose-response can be fitted with support anywhere.
# Pushing the masked arm down into the causal range supplies that support and
# settles the question WITHOUT needing the raw-dimension run.
#
# DESIGN: masked arm only, layer 14, dictionary seed fixed at 42, small k so
# val_EV falls into the causal range. Feeds analyze_capacity_lstruct.py.
#
# COST: 6 SAE fits, ~1 h. GPU.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 8 ]; then
  say "STAGE 8 — capacity extension for overlapping val_EV (masked arm)"
  ck="$HOME/own_sae_data/uniref50_pilot/$MLM/model_final.pt"
  if [ ! -f "$ck" ]; then bad "no checkpoint for $MLM - cannot extend capacity sweep"; else
    for k in 8 16 32; do
      for e in 4 8; do
        cell="outputs_ctrl_cap/${MLM}_k${k}_e${e}/layer_14"
        [ -f "$cell/struct_seq_metrics.csv" ] && { ok "k$k e$e cached"; continue; }
        run "s8_${MLM}_k${k}_e${e}" "$PY" eval_ctrl_plm.py \
          --ckpt "$ck" --name "${MLM}_k${k}_e${e}" --layer 14 \
          --out-root outputs_ctrl_cap --eval-set eval_set \
          --sae-seed 42 --expansion "$e" --k-sparse "$k"
      done
    done
    run "s8_analyze" "$PY" analyze_capacity_lstruct.py --root outputs_ctrl_cap --layer 14 \
      --out results_rigor/capacity_vs_lstruct_extended.csv || true
  fi
  echo "   target: masked val_EV cells landing inside 0.26-0.84, where the causal arm already sits."
fi

# ---------------------------------------------------------------------------
# STAGE 9 — is the 5-permutation null adding material noise?
#
# WHY: L_struct subtracts a null averaged over 5 within-protein permutations.
# Five is few. The permutation SEED is pinned, so the same draws are reused for
# every model and arm and the comparison is paired -- that part is right. What is
# unknown is whether five draws estimate the null tightly enough, and the noise
# will not be symmetric between arms if their activation densities differ.
#
# DESIGN: identical cells, identical seed, only the permutation COUNT changes.
# If the effect moves materially between 5 and 25, every L_struct number in the
# paper carries more null noise than its interval admits.
#
# COST: CPU only, no training. ~30 min.
# ---------------------------------------------------------------------------
if [ "$STAGE" = 0 ] || [ "$STAGE" = 9 ]; then
  say "STAGE 9 — permutation-count sensitivity (5 vs 25 draws)"
  out=results_nshuffle_sensitivity
  mkdir -p "$out"
  for n in 5 25; do
    f="$out/nshuf${n}/contact_def_sweep.csv"
    [ -f "$f" ] && { ok "n=$n cached"; continue; }
    run "s9_nshuf${n}" "$PY" experiment_contact_def_sweep.py \
      --root "$CTRL_REAL" --model-a "$MLM" --model-b "$CLM" \
      --layers "$LAYERS" --n-shuffles "$n" --out "$out/nshuf${n}"
  done
  echo "   compare d and mean_a/mean_b between nshuf5 and nshuf25 at the same cells."
  echo "   a material move means the null is undersampled at 5."
fi

say "summary"
if [ ${#FAILED[@]} -eq 0 ]; then
  ok "all attempted stages completed"
  exit 0
else
  printf '   \033[31m!! %s failure(s):\033[0m\n' "${#FAILED[@]}"
  for f in "${FAILED[@]}"; do echo "      - $f"; done
  echo "   logs: $LOGS/"
  exit 1                # non-zero, so a wrapping queue cannot report OK over a failure
fi
