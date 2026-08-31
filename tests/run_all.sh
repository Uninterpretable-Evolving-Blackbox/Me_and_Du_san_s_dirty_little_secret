#!/usr/bin/env bash
# All smoke tests for the Tier 1 queue. No GPU, no checkpoints, no real data.
#   bash tests/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
rc=0
echo "### syntax"
for f in rescore_denominator.py analyze_top1_agreement.py prep_controlled_corpus.py; do
  $PY -m py_compile "$f" && echo "  [PASS] py_compile $f" || { echo "  [FAIL] py_compile $f"; rc=1; }
done
for f in RUN_TIER1.sh RUN_BLOCKSHUFFLE.sh; do
  bash -n "$f" && echo "  [PASS] bash -n $f" || { echo "  [FAIL] bash -n $f"; rc=1; }
done
echo
echo "### numerics (against the real cpu_stage functions)"
$PY tests/test_rescore_denominator.py 2>&1 | grep -vE "Biopython|warnings.warn|SparseEfficiency" || rc=1
echo
echo "### block shuffle (second destruction procedure)"
$PY tests/test_block_shuffle.py || rc=1
echo
echo "### block-shuffle driver, end to end against stubs"
bash tests/test_run_blockshuffle.sh || rc=1
echo
echo "### top1_share agreement"
bash tests/test_top1_agreement.sh || rc=1
echo
echo "### driver, end to end against stubs"
bash tests/test_run_tier1.sh || rc=1
echo
[ "$rc" -eq 0 ] && echo "=== ALL SUITES PASSED ===" || echo "=== FAILURES ==="
exit $rc
