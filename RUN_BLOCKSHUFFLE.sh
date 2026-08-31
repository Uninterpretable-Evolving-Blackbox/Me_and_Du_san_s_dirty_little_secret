#!/usr/bin/env bash
# RUN_BLOCKSHUFFLE.sh — the second destruction procedure.
#
#   bash RUN_BLOCKSHUFFLE.sh            # all four stages
#   ONLY=1 bash RUN_BLOCKSHUFFLE.sh     # just the corpus
#   bash RUN_BLOCKSHUFFLE.sh --plan     # plan and cost, run nothing
#   SEEDS=42 bash RUN_BLOCKSHUFFLE.sh   # n=1 instead of n=3 (about half the time)
#
# WHY
#   The paper's first stated limitation is that it tests ONE destruction
#   procedure. A second permutation seed does not answer it: composition and
#   length are fixed by construction, so any seed gives a near-identical corpus.
#   A block shuffle is structurally different. It permutes contiguous blocks of
#   K residues, so short-range order survives inside each block and only
#   long-range order is destroyed. If L_struct rises here too, the result is not
#   an artefact of total order destruction. If it does not, that is a boundary
#   on the claim and is worth just as much.
#
# COST  (measured on this box, not estimated)
#   corpus      ~1 h, CPU, streams 3M sequences        once
#   training    0.8 GPU-h per model at 233k tok/s      6 models at n=3 -> ~5 h
#   SAE+metric  125 s per cell                         18 cells -> ~40 min
#   -> about 6 h at n=3, about 3 h at n=1. Not days: a 42M run is 47 minutes.
#
# NOTHING HERE TOUCHES THE EXISTING CORPORA OR CHECKPOINTS. The corpus builder
# refuses to write to the real corpus directory, and every path below carries
# the _blk<K> tag.
set -u
cd "$(dirname "$0")"

PY="${PY:-python}"
ONLY="${ONLY:-}"
[ "${1:-}" = "--plan" ] && PLAN=1 || PLAN=0

BLOCK="${BLOCK:-16}"          # residues per block. 16 keeps roughly one helix turn x4
SEEDS="${SEEDS:-42 43 44}"
DEPTHS="${DEPTHS:-11 14 18}"
TARGET_TOKENS="${TARGET_TOKENS:-6.6e8}"   # 40,283 steps x 32 x 512, the headline budget
BATCH="${BATCH:-32}"; SEQ="${SEQ:-512}"; LR="${LR:-6e-4}"; WARMUP="${WARMUP:-500}"
D_MODEL="${D_MODEL:-320}"; N_HEADS="${N_HEADS:-5}"; N_LAYERS="${N_LAYERS:-30}"
K_SPARSE="${K_SPARSE:-256}"; EXPANSION="${EXPANSION:-8}"; NSHUF="${NSHUF:-5}"

DATA_BLK="${DATA_BLK:-$HOME/own_sae_data/uniref50_pilot_shuf_blk${BLOCK}}"
CKPT_BLK="${CKPT_BLK:-$HOME/own_sae_data/uniref50_pilot_blk${BLOCK}}"
OUT_BLK="${OUT_BLK:-outputs_ctrl_blk${BLOCK}}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/own_sae_data/uniref50_pilot}"

LOGDIR="logs_blockshuffle"; mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S); FAILED=""; RAN=0
_REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

arms_for_seed () { echo "ckpt_mlm_s${1}_token ckpt_clm_s${1}"; }
want () { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
banner () { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }

if [ "$PLAN" = 1 ]; then
  n_models=0; for s in $SEEDS; do n_models=$((n_models+2)); done
  n_cells=$((n_models * $(echo $DEPTHS | wc -w)))
  cat <<PLANEOF
STAGE 1  block-shuffled corpus, block size $BLOCK      ~1 h CPU, once
         -> $DATA_BLK
STAGE 2  pretrain $n_models models at $TARGET_TOKENS tokens   ~0.8 GPU-h each
         -> $CKPT_BLK
STAGE 3  SAE + L_struct, $n_cells cells at $DEPTHS      ~125 s each
         -> $OUT_BLK
STAGE 4  package

seeds: $SEEDS   depths: $DEPTHS   block: $BLOCK
SEEDS=42 halves stages 2 and 3.
PLANEOF
  exit 0
fi

echo "########## RUN_BLOCKSHUFFLE | rev ${_REV} | block ${BLOCK} | started $(date) ##########"

# ------------------------------------------------------------------ preflight
banner "preflight"
_bad=0
for f in prep_controlled_corpus.py train_ctrl_plm.py eval_ctrl_plm.py cpu_stage.py; do
  [ -f "$f" ] || { echo "  !! missing $f"; _bad=1; }
done
if ! $PY -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
  echo "  !! '$PY' does not run"; _bad=1
fi
# The failure that would produce plausible numbers rather than an error: writing
# the block corpus over the real one, or training from the real corpus by mistake.
case "$DATA_BLK" in
  "$HOME/own_sae_data/uniref50_pilot"|"$HOME/own_sae_data/uniref50_pilot_shuf")
    echo "  !! DATA_BLK points at an existing corpus: $DATA_BLK"
    echo "     That would overwrite results the paper depends on. Refusing."
    _bad=1;;
esac
case "$CKPT_BLK" in
  "$CKPT_ROOT"|"$HOME/own_sae_data/uniref50_pilot_shuf")
    echo "  !! CKPT_BLK collides with an existing checkpoint tree. Refusing."; _bad=1;;
esac
[ -d "$HOME/own_sae_data" ] || { echo "  !! no $HOME/own_sae_data"; _bad=1; }
[ "$_bad" -ne 0 ] && { echo "preflight FAILED — nothing run"; exit 1; }
echo "  paths distinct from the real corpus and checkpoints, OK"

# =========================================================== STAGE 1  corpus
if want 1; then
  banner "STAGE 1  block-shuffled corpus (block $BLOCK)"
  if [ -f "$DATA_BLK/meta.json" ]; then
    echo "  [done] $DATA_BLK"
    RAN=$((RAN+1))
  else
    log="$LOGDIR/s1_corpus_${STAMP}.log"
    if $PY -u prep_controlled_corpus.py --block-shuffle "$BLOCK" > "$log" 2>&1; then
      grep -E "BLOCK-SHUFFLED|kept|tokens" "$log" | tail -4 | sed 's/^/    /'
      RAN=$((RAN+1))
    else
      echo "  !! FAILED (see $log)"; tail -8 "$log" | sed 's/^/     /'
      FAILED="$FAILED corpus"
    fi
  fi
  [ -f "$DATA_BLK/meta.json" ] || { echo "  no corpus — later stages cannot run"; exit 1; }
fi

# =========================================================== STAGE 2  training
if want 2; then
  banner "STAGE 2  pretraining on the block-shuffled corpus"
  [ -f "$DATA_BLK/meta.json" ] || { echo "  !! no $DATA_BLK — run stage 1 first"; exit 1; }
  n=0
  for s in $SEEDS; do
    for obj in mlm clm; do
      case "$obj" in mlm) arm="ckpt_mlm_s${s}_token";; clm) arm="ckpt_clm_s${s}";; esac
      out="$CKPT_BLK/$arm"
      if [ -f "$out/model_final.pt" ]; then echo "  [done] $arm"; n=$((n+1)); continue; fi
      mkdir -p "$out"
      res=""; [ -f "$out/model_resume.pt" ] && res="--resume $out/model_resume.pt"
      echo "=== [train] $arm $(date +%H:%M:%S) ==="
      log="$LOGDIR/s2_${arm}_${STAMP}.log"
      if $PY -u train_ctrl_plm.py --objective "$obj" --data-dir "$DATA_BLK" \
             --out-dir "$out" --seed "$s" --target-tokens "$TARGET_TOKENS" \
             --batch-size "$BATCH" --seq-len "$SEQ" --lr "$LR" --warmup "$WARMUP" \
             --rolling-resume --d-model "$D_MODEL" --n-heads "$N_HEADS" \
             --n-layers "$N_LAYERS" $res > "$log" 2>&1; then
        tail -3 "$log" | sed 's/^/    /'; n=$((n+1))
      else
        echo "  !! FAILED (see $log)"; tail -8 "$log" | sed 's/^/     /'
        FAILED="$FAILED train:$arm"
      fi
    done
  done
  [ "$n" -eq 0 ] && { echo "  NO models produced. Not a silent skip."; FAILED="$FAILED stage2-empty"; } \
                 || { echo "  [STAGE 2] $n model(s)"; RAN=$((RAN+1)); }
fi

# =========================================================== STAGE 3  metric
if want 3; then
  banner "STAGE 3  SAE + L_struct at $DEPTHS"
  n=0
  for s in $SEEDS; do
    for arm in $(arms_for_seed "$s"); do
      ck="$CKPT_BLK/$arm/model_final.pt"
      [ -f "$ck" ] || { echo "  [skip] no $ck"; continue; }
      for L in $DEPTHS; do
        dest="$OUT_BLK/$arm/layer_$L"
        [ -f "$dest/struct_seq_metrics.csv" ] && { echo "  [done] $arm L$L"; n=$((n+1)); continue; }
        echo "=== [metric] $arm L$L $(date +%H:%M:%S) ==="
        log="$LOGDIR/s3_${arm}_L${L}_${STAMP}.log"
        if $PY -u eval_ctrl_plm.py --ckpt "$ck" --name "$arm" --layer "$L" \
               --out-root "$OUT_BLK" --eval-set eval_set \
               --k-sparse "$K_SPARSE" --expansion "$EXPANSION" > "$log" 2>&1 &&
           $PY -u cpu_stage.py --layer-dir "$dest" --model-type residue \
               --n-shuffles "$NSHUF" >> "$log" 2>&1; then
          grep -E "val_EV|DEGENERATE" "$log" | tail -1 | sed 's/^/    /'; n=$((n+1))
        else
          echo "  !! FAILED (see $log)"; tail -6 "$log" | sed 's/^/     /'
          FAILED="$FAILED metric:${arm}/L${L}"
        fi
      done
    done
  done
  [ "$n" -eq 0 ] && { echo "  NO cells produced. Not a silent skip."; FAILED="$FAILED stage3-empty"; } \
                 || { echo "  [STAGE 3] $n cell(s)"; RAN=$((RAN+1)); }
fi

# ------------------------------------------------------------------ package
banner "packaging"
PKG="blockshuffle_blk${BLOCK}_$(date +%Y%m%d)"
rm -rf "$PKG"; mkdir -p "$PKG"
find "$OUT_BLK" -name struct_seq_metrics.csv -print0 2>/dev/null |
  while IFS= read -r -d '' f; do mkdir -p "$PKG/$(dirname "$f")"; cp "$f" "$PKG/$f"; done
[ -f "$DATA_BLK/meta.json" ] && { mkdir -p "$PKG/corpus"; cp "$DATA_BLK/meta.json" "$PKG/corpus/"; }
mkdir -p "$PKG/logs"; cp "$LOGDIR"/*_"$STAMP".log "$PKG/logs/" 2>/dev/null
git rev-parse HEAD > "$PKG/git_revision.txt" 2>/dev/null
git status --short > "$PKG/git_status.txt" 2>/dev/null
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"
elif command -v shasum >/dev/null 2>&1; then SHA="shasum -a 256"; else SHA=""; fi
[ -n "$SHA" ] && ( cd "$PKG" && find . -type f ! -name SHA256SUMS.txt -exec $SHA {} + > SHA256SUMS.txt 2>/dev/null )

{
  echo "# Block-shuffle batch (block size $BLOCK)"
  echo
  echo "- Date: $(date +%Y-%m-%d)"
  echo "- Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "- Seeds: $SEEDS | Depths: $DEPTHS | tokens: $TARGET_TOKENS | n_shuffles: $NSHUF"
  echo "- Metric settings unchanged: 8 A, |i-j| >= 12, top-10%"
  echo
  echo "## Mean L_struct per cell"
  echo
  echo "| arm | layer | mean L_struct |"
  echo "|---|---:|---:|"
  find "$OUT_BLK" -name struct_seq_metrics.csv 2>/dev/null | sort | while read -r f; do
    arm=$(echo "$f" | awk -F/ '{print $2}'); lay=$(echo "$f" | awk -F/ '{print $3}')
    m=$($PY - "$f" <<'PYEOF'
import csv, sys, statistics as st
with open(sys.argv[1]) as fh:
    print(f"{st.mean(float(r['struct_delta']) for r in csv.DictReader(fh)):+.5f}")
PYEOF
)
    echo "| $arm | ${lay#layer_} | $m |"
  done
  echo
  echo "Compare against outputs_ctrl (native) and outputs_ctrl_shuf (fully order-destroyed)"
  echo "at the same arms and depths. The question is whether L_struct rises here too."
  [ -n "$FAILED" ] && { echo; echo "## FAILED"; echo; echo "$FAILED"; }
} > "$PKG/RESULT_SUMMARY.md"

tar czf "${PKG}.tgz" "$PKG"
echo "  wrote ${PKG}.tgz  ($(du -h "${PKG}.tgz" | cut -f1))"
echo
if [ -n "$FAILED" ]; then echo "FAILURES:$FAILED"; echo "Send the archive anyway."; exit 1; fi
[ "$RAN" -eq 0 ] && { echo "NOTHING RAN — check ONLY= and the preflight."; exit 1; }
echo "All stages OK. Send ${PKG}.tgz."
