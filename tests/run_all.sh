#!/usr/bin/env bash
# All smoke tests for the Tier 1 queue. No GPU, no checkpoints, no real data.
#   bash tests/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
rc=0
echo "### syntax"
for f in rescore_denominator.py analyze_top1_agreement.py; do
  $PY -m py_compile "$f" && echo "  [PASS] py_compile $f" || { echo "  [FAIL] py_compile $f"; rc=1; }
done
bash -n RUN_TIER1.sh && echo "  [PASS] bash -n RUN_TIER1.sh" || { echo "  [FAIL] bash -n RUN_TIER1.sh"; rc=1; }
echo
echo "### numerics (against the real cpu_stage functions)"
$PY tests/test_rescore_denominator.py 2>&1 | grep -vE "Biopython|warnings.warn|SparseEfficiency" || rc=1
echo
echo "### top1_share agreement"
bash tests/test_top1_agreement.sh || rc=1
echo
echo "### driver, end to end against stubs"
bash tests/test_run_tier1.sh || rc=1
echo
[ "$rc" -eq 0 ] && echo "=== ALL SUITES PASSED ===" || echo "=== FAILURES ==="
exit $rc
