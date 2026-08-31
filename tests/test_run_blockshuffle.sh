#!/usr/bin/env bash
# Stub-driven smoke test for RUN_BLOCKSHUFFLE.sh. No GPU, no corpus, no network.
# The check that matters: it must refuse to write over the real corpus or the
# real checkpoints, because that failure destroys data the paper depends on.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fails=0
ck(){ if [ "$2" = "$3" ]; then echo "  [PASS] $1"; else echo "  [FAIL] $1 (got '$2' want '$3')"; fails=$((fails+1)); fi; }
ckc(){ if grep -q "$2" "$3"; then echo "  [PASS] $1"; else echo "  [FAIL] $1 (no '$2')"; fails=$((fails+1)); fi; }

mkdir -p "$T/bin" "$T/home/own_sae_data"
cat > "$T/bin/python" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = "-" ] && { shift; exec python3 - "$@"; }
[ "${1:-}" = "-c" ] && exit 0
[ "${1:-}" = "-u" ] && shift
s="$(basename "${1:-}")"; shift
get(){ local k="$1"; shift; while [ $# -gt 0 ]; do [ "$1" = "$k" ] && { echo "$2"; return; }; shift; done; }
case "$s" in
  prep_controlled_corpus.py)
    b="$(get --block-shuffle "$@")"; d="$HOME/own_sae_data/uniref50_pilot_shuf_blk${b}"
    mkdir -p "$d"; echo '{"n_sequences":3000000}' > "$d/meta.json"
    echo "!! BLOCK-SHUFFLED CORPUS, block size $b"; exit 0 ;;
  train_ctrl_plm.py)
    o="$(get --out-dir "$@")"; mkdir -p "$o"; : > "$o/model_final.pt"; echo "done"; exit 0 ;;
  eval_ctrl_plm.py)
    r="$(get --out-root "$@")"; n="$(get --name "$@")"; L="$(get --layer "$@")"
    mkdir -p "$r/$n/layer_$L"; : > "$r/$n/layer_$L/Z.npy"; echo "val_EV 0.91"; exit 0 ;;
  cpu_stage.py)
    d="$(get --layer-dir "$@")"
    printf 'feature_idx,struct_delta\n0,0.027\n1,0.029\n' > "$d/struct_seq_metrics.csv"; exit 0 ;;
  *) echo "stub: unexpected $s" >&2; exit 99 ;;
esac
STUB
chmod +x "$T/bin/python"

mkfix(){ rm -rf "$T/w"; mkdir -p "$T/w"
  for f in RUN_BLOCKSHUFFLE.sh prep_controlled_corpus.py train_ctrl_plm.py eval_ctrl_plm.py cpu_stage.py; do
    cp "$REPO/$f" "$T/w/$f"; done
  mkdir -p "$T/home/own_sae_data"; }
run(){ ( cd "$T/w" && env PATH="$T/bin:$PATH" HOME="$T/home" PY=python SEEDS=42 "$@" bash RUN_BLOCKSHUFFLE.sh ); }

echo "== RUN_BLOCKSHUFFLE.sh =="

# the guard that protects irreplaceable data
mkfix; run DATA_BLK="$T/home/own_sae_data/uniref50_pilot" > "$T/l" 2>&1
ck "refuses to overwrite the real corpus" "$?" "1"
ckc "  and says why" "results the paper depends on" "$T/l"

mkfix; run CKPT_BLK="$T/home/own_sae_data/uniref50_pilot" > "$T/l" 2>&1
ck "refuses to collide with the real checkpoints" "$?" "1"

mkfix; run DATA_BLK="$T/home/own_sae_data/uniref50_pilot_shuf" > "$T/l" 2>&1
ck "refuses to overwrite the residue-shuffled corpus" "$?" "1"

# happy path, n=1: 1 corpus, 2 models, 6 cells
mkfix; run > "$T/l" 2>&1
ck "full run exits 0" "$?" "0"
ck "builds the corpus" "$(test -f "$T/home/own_sae_data/uniref50_pilot_shuf_blk16/meta.json" && echo y || echo n)" "y"
ck "trains both arms" "$(ls "$T/home/own_sae_data/uniref50_pilot_blk16" | wc -l | tr -d ' ')" "2"
ck "scores 6 cells" "$(find "$T/w/outputs_ctrl_blk16" -name struct_seq_metrics.csv | wc -l | tr -d ' ')" "6"
ckc "  summary carries a computed mean" "+0.02800" "$T/w/blockshuffle_blk16_$(date +%Y%m%d)/RESULT_SUMMARY.md"
ckc "  summary says what to compare against" "outputs_ctrl_shuf" "$T/w/blockshuffle_blk16_$(date +%Y%m%d)/RESULT_SUMMARY.md"

# resume
run > "$T/l2" 2>&1; ck "second run exits 0" "$?" "0"
ck "resume retrains nothing" "$(grep -c '=== \[train\]' "$T/l2")" "0"
ck "resume rescores nothing" "$(grep -c '=== \[metric\]' "$T/l2")" "0"

# block size flows through to every path
mkfix; run BLOCK=8 > "$T/l" 2>&1
ck "BLOCK=8 uses its own corpus dir" "$(test -f "$T/home/own_sae_data/uniref50_pilot_shuf_blk8/meta.json" && echo y || echo n)" "y"
ck "BLOCK=8 uses its own output dir" "$(test -d "$T/w/outputs_ctrl_blk8" && echo y || echo n)" "y"

mkfix; run ONLY=9 > "$T/l" 2>&1
ck "unknown ONLY exits non-zero" "$?" "1"
ckc "  and says nothing ran" "NOTHING RAN" "$T/l"

echo "--"
[ "$fails" -eq 0 ] && { echo "All checks passed."; exit 0; }
echo "FAILED: $fails"; exit 1
