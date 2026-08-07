#!/usr/bin/env bash
# preflight.sh — run this BEFORE committing a box to the long jobs.
#
# Written to be run by an agent: every check prints PASS / FAIL / SKIP with the
# expected value stated, and the script exits non-zero if anything FAILed. No
# check here requires judgement.
#
# THE RULE: on any FAIL, stop and report. Do not repair and continue. Every
# check below exists because the thing it guards has already failed silently at
# least once in this project, and a silent failure here produces plausible
# numbers rather than a crash -- which is far more expensive to catch later.
#
#   ./preflight.sh env         # before anything.            ~1 min, no GPU
#   ./preflight.sh data        # before jobs 1-2 (metrics).  ~1 min
#   ./preflight.sh ckpt        # before job 3 (attack).      ~2 min
#   ./preflight.sh post-setup  # after `RUN_INTERPLM_STRESS.sh setup`
#   ./preflight.sh first-cell  # after the FIRST grid cell finishes
#   ./preflight.sh all         # env + data + ckpt
#
set -u
PASS=0; FAIL=0; SKIP=0
ok()   { echo "  PASS  $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
skip() { echo "  SKIP  $*"; SKIP=$((SKIP+1)); }
hdr()  { echo; echo "== $* =="; }

PY=${PY:-$PWD/.venv/bin/python}
CKPT_ROOT=${CKPT_ROOT:-$HOME/own_sae_data/uniref50_pilot}
CKPT_ROOT_SHUF=${CKPT_ROOT_SHUF:-$HOME/own_sae_data/uniref50_pilot_shuf}
BASE=${BASE:-$HOME/interplm_stress}
SHA=$(command -v sha256sum >/dev/null && echo sha256sum || echo "shasum -a 256")

# ---------------------------------------------------------------- env
check_env () {
  hdr "environment"
  [ -x "$PY" ] && ok "interpreter exists: $PY" || bad "no interpreter at $PY (set PY=...)"
  if [ -x "$PY" ]; then
    v=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
    [ -n "$v" ] && ok "python $v" || bad "interpreter will not run"
    "$PY" -c 'import numpy,pandas,scipy,sklearn' 2>/dev/null \
      && ok "numpy/pandas/scipy/sklearn import" || bad "a core package is missing"
    g=$("$PY" -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)
    [ "$g" = "True" ] && ok "CUDA visible" || skip "CUDA not visible (fine for jobs 1-2, NOT for 3-4)"
  fi
  free=$(df -Pg "$HOME" 2>/dev/null | awk 'NR==2{print $4}')
  if [ -n "${free:-}" ]; then
    [ "$free" -ge 80 ] && ok "free disk ${free} GB (need >=60 for the attack)" \
                       || bad "free disk ${free} GB — the attack needs >=60 GB and will die mid-grid"
  else skip "could not read free disk"; fi
  # the PY quoting regression that killed all 13 stage-11 cells last time
  if grep -q 'PY="${PY:-$PY_BIN -u}"' RUN_MUSTRUNS.sh 2>/dev/null; then
    bad "RUN_MUSTRUNS.sh still has the PY='...python -u' quoting bug (line ~79). Override with PY=<interpreter> or the stages exec a file named 'python -u'"
  else ok "no PY quoting regression in RUN_MUSTRUNS.sh"; fi
}

# ---------------------------------------------------------------- data
check_data () {
  hdr "inputs for jobs 1-2 (no-model baseline, extra metrics)"
  for f in cache/scope_40.fa cache/residue_features.csv; do
    [ -s "$f" ] && ok "$f present" || bad "$f missing — jobs 1-2 cannot run"
  done
  n=$(find outputs_ctrl -name Z.npy 2>/dev/null | wc -l | tr -d ' ')
  if [ "${n:-0}" -gt 0 ]; then ok "Z.npy present in $n layer dir(s)"
  else bad "no Z.npy under outputs_ctrl — run 'STAGE=1 bash RUN_MUSTRUNS.sh' (~2 h) first"; fi

  # The synthetic layer has invariants that hold regardless of which source dir
  # it was built from. If these are wrong the artefact is malformed and every
  # downstream 'no-model' number is meaningless.
  if [ -f outputs_synthetic/composition/Z.npy ]; then
    "$PY" - <<'EOF' && ok "synthetic layer invariants (29 features, none dead, one all-ones column)" || bad "synthetic layer malformed — rebuild with make_synthetic_layer.py"
import numpy as np, sys, json
Z = np.load("outputs_synthetic/composition/Z.npy", mmap_mode="r")
occ = (np.asarray(Z) > 0).mean(0)
assert Z.shape[1] == 29, f"expected 29 features, got {Z.shape[1]}"
assert (occ > 0).all(), "a synthetic feature never fires"
assert np.isclose(occ.max(), 1.0), "no all-ones TRIVIAL column"
L = np.load("outputs_synthetic/composition/lengths.npy")
assert int(L.sum()) == Z.shape[0], "sum(lengths) != Z rows"
EOF
  else skip "outputs_synthetic/composition not built yet (job 1 builds it)"; fi
}

# ---------------------------------------------------------------- checkpoints
check_ckpt () {
  hdr "checkpoints for job 3 (InterPLM attack)"
  [ "$CKPT_ROOT" = "$CKPT_ROOT_SHUF" ] \
    && bad "CKPT_ROOT == CKPT_ROOT_SHUF. The shuffled arms would be the real models and the output would read 'shuffled == real' — indistinguishable from the metric PASSING its validity check. This is the single most damaging failure available here." \
    || ok "CKPT_ROOT and CKPT_ROOT_SHUF are different paths"

  miss=0
  for c in "$CKPT_ROOT"/ckpt_mlm_s4{2,3,4}_token "$CKPT_ROOT"/ckpt_clm_s4{2,3,4} \
           "$CKPT_ROOT_SHUF"/ckpt_mlm_s42_token "$CKPT_ROOT_SHUF"/ckpt_clm_s42; do
    [ -f "$c/model_final.pt" ] || { echo "        missing: $c/model_final.pt"; miss=$((miss+1)); }
  done
  [ "$miss" -eq 0 ] && ok "all 8 checkpoints present" || bad "$miss checkpoint(s) missing"

  # Distinctness by CONTENT, not path. Copies and symlinks pass a path check.
  if [ "$miss" -eq 0 ]; then
    d=$(for c in "$CKPT_ROOT"/ckpt_mlm_s4{2,3,4}_token "$CKPT_ROOT"/ckpt_clm_s4{2,3,4} \
                 "$CKPT_ROOT_SHUF"/ckpt_mlm_s42_token "$CKPT_ROOT_SHUF"/ckpt_clm_s42; do
          $SHA "$c/model_final.pt" | cut -d' ' -f1; done | sort | uniq -d)
    [ -z "$d" ] && ok "8 checkpoints, all distinct by sha256" \
                || bad "two or more checkpoints are byte-identical — see the CKPT_ROOT note above"
  fi

  # The 500 tok/param models are a DIFFERENT budget. Mixing them in silently
  # would compare 21e9-token models against 660e6-token ones.
  if [ -f "$CKPT_ROOT/ckpt_clm_s42/model_final.pt" ]; then
    t=$("$PY" -c "import torch;print(torch.load('$CKPT_ROOT/ckpt_clm_s42/model_final.pt',map_location='cpu',weights_only=False).get('tokens_seen','?'))" 2>/dev/null)
    case "$t" in
      65999*|66000*) ok "CKPT_ROOT is the 15.7 tok/param tree (tokens_seen=$t)" ;;
      209999*|21000*) bad "CKPT_ROOT points at the 500 tok/param models (tokens_seen=$t) — wrong tree for this grid" ;;
      *) skip "could not read tokens_seen (got '$t')" ;;
    esac
  fi
}

# ---------------------------------------------------------------- post-setup
check_post_setup () {
  hdr "after 'RUN_INTERPLM_STRESS.sh setup'"
  REPO=$BASE/interplm_repo
  T=$REPO/examples/train_ctrl_sae.py
  if [ ! -f "$T" ]; then bad "$T not generated — setup did not complete"; return; fi
  for pat in CTRL_EMBD_DIR CTRL_SAVE_DIR CTRL_DIM CTRL_L1 EvaluationConfig 'seed=int(os.environ'; do
    grep -q "$pat" "$T" && ok "generated trainer has $pat" \
      || bad "generated trainer is MISSING $pat — it would silently run with THEIR hyperparameters, not ours"
  done
  grep -q 'CTRL_EXP' "$T" && ok "expansion is parameterised" || bad "expansion not parameterised"
  # their file must be untouched
  if [ -f "$REPO/examples/train_basic_sae.py" ]; then
    ( cd "$REPO" && git diff --quiet -- examples/train_basic_sae.py 2>/dev/null ) \
      && ok "InterPLM's train_basic_sae.py is unmodified" \
      || bad "InterPLM's own file was edited — the 'their code, unmodified' claim no longer holds"
  fi
  n=$(ls "$BASE"/ann/processed/valid 2>/dev/null | wc -l | tr -d ' ')
  [ "${n:-0}" -gt 0 ] && ok "annotation valid split is non-empty" || bad "ann/processed/valid is empty — extract_annotations or prepare_eval_set failed"
}

# ---------------------------------------------------------------- first cell
check_first_cell () {
  hdr "after the FIRST grid cell"
  R=$BASE/results
  [ -s "$R/sae_quality.txt" ] && ok "sae_quality.txt exists" \
    || bad "no sae_quality.txt — dictionary quality must be written BEFORE any concept number"
  if [ -s "$R/sae_quality.txt" ] && [ -s "$R/concept_f1.txt" ]; then
    [ "$R/sae_quality.txt" -ot "$R/concept_f1.txt" ] \
      && ok "quality was written before the concept score (outcome-blind order held)" \
      || bad "concept_f1.txt is older than sae_quality.txt — selection order violated"
  fi

  # THE INERT-SEED CHECK. InterPLM's dataloader calls torch.manual_seed(config.seed)
  # and is built before the SAE, so a seed injected anywhere else is overwritten.
  # That bug already produced three bit-identical 'seeds' in this project's own
  # sae_quality.txt (s0/s1/s2 identical in both arms). If it recurs, every
  # 'three SAE seeds' claim is actually n=1.
  first=$(ls -d "$BASE"/interplm_repo/models/grid/*_s0 2>/dev/null | head -1)
  if [ -n "${first:-}" ]; then
    stem=${first%_s0}
    h=$(for s in 0 1 2; do [ -f "${stem}_s${s}/ae.pt" ] && $SHA "${stem}_s${s}/ae.pt" | cut -d' ' -f1; done)
    n_uniq=$(echo "$h" | sort -u | grep -c .)
    n_tot=$(echo "$h" | grep -c .)
    if [ "$n_tot" -lt 2 ]; then skip "fewer than 2 seeds finished yet"
    elif [ "$n_uniq" -eq "$n_tot" ]; then ok "$n_tot SAE seeds produce distinct dictionaries"
    else bad "SAE seeds are INERT ($n_uniq distinct of $n_tot). Every 'three seeds' number would be n=1. Check that DataloaderConfig(seed=...) took effect."; fi
  else skip "no completed seed-0 cell yet"; fi

  if [ -s "$R/sae_quality.txt" ]; then
    echo "        --- live-feature counts so far (read these before any F1) ---"
    tail -6 "$R/sae_quality.txt" | sed 's/^/        /'
    echo "        If the masked and causal arms differ greatly here, the objective"
    echo "        contrast is NOT interpretable and the non-comparability is itself"
    echo "        the finding. Report it rather than the F1 gap."
  fi
}

case "${1:-all}" in
  env) check_env ;;
  data) check_data ;;
  ckpt) check_ckpt ;;
  post-setup) check_post_setup ;;
  first-cell) check_first_cell ;;
  all) check_env; check_data; check_ckpt ;;
  *) echo "usage: $0 {env|data|ckpt|post-setup|first-cell|all}"; exit 2 ;;
esac

echo
echo "== $PASS passed, $FAIL failed, $SKIP skipped =="
[ "$FAIL" -eq 0 ] || { echo "STOP. Do not start the run. Report the FAIL lines above."; exit 1; }
echo "OK to proceed."
