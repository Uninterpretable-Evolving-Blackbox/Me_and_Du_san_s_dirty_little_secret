#!/usr/bin/env bash
# ============================================================================
# RUN_MUSTRUNS.sh — the outstanding experiments, in value order.
#
#   bash RUN_MUSTRUNS.sh            # stages 1-4 (no training; hours, not days)
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
    run "s1_${tag}" python experiment_contact_def_sweep.py \
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
    run s2_aa_selectivity python experiment_aa_selectivity.py \
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
      run "s3_${arm}" python measure_rank_ev.py \
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
    run s4_mlp_probe python eval_ctrl_saefree.py --mlp \
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
    run s5_prep python prep_controlled_corpus.py --shuffle-residues \
      --out-dir "$SHUF_DATA"
  else ok "shuffled corpus present"; fi
  for s in 43 44; do
    for obj in mlm clm; do
      out="$HOME/own_sae_data/uniref50_pilot_shuf/ckpt_${obj}_s${s}"
      [ -d "$out" ] && { ok "${obj}_s${s} cached"; continue; }
      run "s5_${obj}_s${s}" python train_ctrl_plm.py \
        --data-dir "$SHUF_DATA" --objective "$obj" --seed "$s" \
        --data-order-seed 1234 --out-dir "$out"
    done
  done
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
