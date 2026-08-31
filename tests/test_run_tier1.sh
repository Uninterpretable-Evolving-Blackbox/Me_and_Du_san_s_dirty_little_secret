#!/usr/bin/env bash
# End-to-end smoke test for RUN_TIER1.sh against a stubbed tree.
# No GPU, no checkpoints, no real activations. Exercises: preflight failures,
# stage selection, resume, empty-stage-is-a-failure, cell failure, packaging,
# and the RESULT_SUMMARY table (whose embedded python really runs).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fails=0
ck(){ if [ "$2" = "$3" ]; then echo "  [PASS] $1"; else echo "  [FAIL] $1 (got '$2' want '$3')"; fails=$((fails+1)); fi; }
ckc(){ if grep -q "$2" "$3"; then echo "  [PASS] $1"; else echo "  [FAIL] $1 (no '$2')"; fails=$((fails+1)); fi; }

# ---------------------------------------------------------------- stub python
mkdir -p "$T/bin"
cat > "$T/bin/python" <<'STUB'
#!/usr/bin/env bash
# Dispatching stub. Honours the real contract of each script it fakes.
[ "${1:-}" = "-" ] && { shift; exec python3 - "$@"; }        # packaging heredoc
[ "${1:-}" = "-c" ] && { [ "${STUB_NO_CPUSTAGE:-0}" = 1 ] && { echo "ModuleNotFoundError: No module named 'cpu_stage'" >&2; exit 1; }; exit 0; }
[ "${1:-}" = "-u" ] && shift
script="$(basename "${1:-}")"; shift
get(){ local k="$1"; shift; while [ $# -gt 0 ]; do [ "$1" = "$k" ] && { echo "$2"; return; }; shift; done; }
case "$script" in
  analyze_top1_agreement.py)
    echo "VERDICT: AGREE on 2 cell(s)."; echo "  check count stays at six"; exit 0 ;;
  rescore_denominator.py)
    d="$(get --layer-dir "$@")"
    [ "${STUB_FAIL_DENOM:-}" = "$d" ] && { echo "SELF-CHECK FAILED: max diff 3.1e-02" >&2; exit 1; }
    echo "  self-check max|struct_delta - struct_delta_sd| = 0.000e+00"
    echo "  self-check PASSED (max diff 0.000e+00)"
    printf 'feature_idx,struct_delta_sd,struct_delta_fixed,struct_delta_iqr,struct_delta_rank\n' > "$d/struct_seq_metrics_denominators.csv"
    printf '0,0.0180,0.0900,0.0300,0.0120\n1,0.0200,0.1100,0.0340,0.0140\n' >> "$d/struct_seq_metrics_denominators.csv"
    exit 0 ;;
  experiment_interplm_metric.py)
    o="$(get --out "$@")"; mkdir -p "$(dirname "$o")"
    printf 'feature,d_struct,p\n0,0.44,0.01\n' > "$o"; echo "  mean d_struct 0.44"; exit 0 ;;
  eval_ctrl_plm.py)
    r="$(get --out-root "$@")"; n="$(get --name "$@")"; L="$(get --layer "$@")"
    mkdir -p "$r/$n/layer_$L"; : > "$r/$n/layer_$L/Z.npy"; exit 0 ;;
  *) echo "stub: unexpected script '$script'" >&2; exit 99 ;;
esac
STUB
chmod +x "$T/bin/python"

# ---------------------------------------------------------------- fake repo
mkfixture(){
  rm -rf "$T/w"; mkdir -p "$T/w"
  for f in RUN_TIER1.sh rescore_denominator.py analyze_top1_agreement.py \
           cpu_stage.py experiment_interplm_metric.py eval_ctrl_plm.py; do
    cp "$REPO/$f" "$T/w/$f"
  done
  for root in outputs_ctrl outputs_ctrl_shuf; do
    for arm in ckpt_mlm_s42_token ckpt_clm_s42; do
      for L in 11 14 18; do
        mkdir -p "$T/w/$root/$arm/layer_$L"
        : > "$T/w/$root/$arm/layer_$L/Z.npy"
        printf 'feature_idx,struct_delta\n0,0.018\n' > "$T/w/$root/$arm/layer_$L/struct_seq_metrics.csv"
      done
    done
  done
  mkdir -p "$T/w/ck/ckpt_mlm_s42_token" "$T/w/ck/ckpt_clm_s42"
  : > "$T/w/ck/ckpt_mlm_s42_token/model_final.pt"
  : > "$T/w/ck/ckpt_clm_s42/model_final.pt"
}
# NB: env, not a bare assignment prefix — words that come out of "$@" are
# not re-scanned as assignments by any POSIX shell.
run(){ ( cd "$T/w" && env PATH="$T/bin:$PATH" PY=python CKPT_ROOT="$T/w/ck" SEEDS=42 "$@" bash RUN_TIER1.sh ) ; }

echo "== RUN_TIER1.sh =="

# 1. preflight: missing script
mkfixture; rm "$T/w/rescore_denominator.py"
run > "$T/l" 2>&1; ck "preflight fails on a missing script" "$?" "1"
ckc "  and names it" "missing rescore_denominator.py" "$T/l"
ckc "  and runs nothing" "nothing run" "$T/l"

# 2. preflight: cpu_stage not importable, stage 2 requested
mkfixture
run STUB_NO_CPUSTAGE=1 ONLY=2 > "$T/l" 2>&1; ck "preflight fails when cpu_stage will not import" "$?" "1"
ckc "  and shows the import error" "cannot run" "$T/l"

# 3. preflight: no Z.npy anywhere
mkfixture; find "$T/w/outputs_ctrl" "$T/w/outputs_ctrl_shuf" -name Z.npy -delete
run ONLY=2 > "$T/l" 2>&1; ck "preflight fails when no cell has Z.npy" "$?" "1"
ckc "  and gives the rebuild command" "RUN_MUSTRUNS.sh" "$T/l"

# 4. happy path, stage 1
mkfixture; run ONLY=1 > "$T/l" 2>&1
ck "stage 1 exits 0" "$?" "0"
ckc "  and surfaces the verdict" "VERDICT: AGREE" "$T/l"

# 5. stage 2 writes one denominator CSV per cell (2 roots x 1 seed x 2 arms x 3 depths = 12)
mkfixture; run ONLY=2 > "$T/l" 2>&1
ck "stage 2 exits 0" "$?" "0"
# count only the source trees: the package dir holds a second copy of each
ck "stage 2 covers every cell" "$(find "$T/w/outputs_ctrl" "$T/w/outputs_ctrl_shuf" -name struct_seq_metrics_denominators.csv | wc -l | tr -d ' ')" "12"
ck "  and packaging copies them all" "$(find "$T/w/tier1_results_$(date +%Y%m%d)" -name struct_seq_metrics_denominators.csv | wc -l | tr -d ' ')" "12"
ckc "  and reports the self-check" "self-check PASSED" "$T/l"

# 6. resume: a second run re-does nothing
run ONLY=2 > "$T/l2" 2>&1
ck "second run exits 0" "$?" "0"
ck "resume skips all 12 cells" "$(grep -c '\[done\]' "$T/l2")" "12"
ck "resume re-runs none" "$(grep -c '=== \[denom\]' "$T/l2")" "0"

# 7. a failing cell -> non-zero exit, archive still written
mkfixture
run ONLY=2 STUB_FAIL_DENOM=outputs_ctrl/ckpt_clm_s42/layer_14 > "$T/l" 2>&1
ck "a failed cell exits non-zero" "$?" "1"
ckc "  and names the cell" "denom:ckpt_clm_s42/L14" "$T/l"
ckc "  and still says to send the archive" "Send the archive anyway" "$T/l"
ck "  and the archive exists" "$(ls "$T/w"/tier1_results_*.tgz 2>/dev/null | wc -l | tr -d ' ')" "1"
ckc "  and RESULT_SUMMARY records the failure" "## FAILED" "$T/w/tier1_results_$(date +%Y%m%d)/RESULT_SUMMARY.md"

# 8. empty stage is a failure, not a green tick
mkfixture; find "$T/w/outputs_ctrl" "$T/w/outputs_ctrl_shuf" -name struct_seq_metrics.csv -delete
run ONLY=2 > "$T/l" 2>&1; ck "a stage that produces nothing exits non-zero" "$?" "1"
ckc "  and says so explicitly" "NO cells produced" "$T/l"

# 9. stage 3: 2 roots x 2 arms x 3 depths x 2 gates = 24
mkfixture; run ONLY=3 > "$T/l" 2>&1
ck "stage 3 exits 0" "$?" "0"
ck "stage 3 covers every gate x cell" "$(ls "$T/w/results_interplm_metric" | grep -c 'gate-')" "24"
ck "  both gate modes present" "$(ls "$T/w/results_interplm_metric" | grep -c 'gate-raw')" "12"

# 10. stage 4 builds the randomised layer dirs then scores them
mkfixture; run ONLY=4 > "$T/l" 2>&1
ck "stage 4 exits 0" "$?" "0"
ck "stage 4 embeds the randomised models" "$(find "$T/w/outputs_ctrl_randominit_s42" -name Z.npy | wc -l | tr -d ' ')" "6"
ck "stage 4 scores them at both gates" "$(ls "$T/w/results_interplm_metric" | grep -c '^randominit_')" "12"

# 11. packaging: summary table really renders (its embedded python runs for real)
mkfixture; run ONLY=2 > /dev/null 2>&1
S="$T/w/tier1_results_$(date +%Y%m%d)/RESULT_SUMMARY.md"
ckc "summary has the denominator table header" "| condition | arm | layer | sd | fixed | iqr | rank |" "$S"
ck "summary has one row per cell" "$(grep -c '^| outputs_ctrl' "$S")" "12"
ckc "summary carries a real computed mean" "+0.01900" "$S"
ckc "summary records the git revision" "Commit:" "$S"
ck "checksums written" "$(test -s "$T/w/tier1_results_$(date +%Y%m%d)/SHA256SUMS.txt" && echo yes || echo no)" "yes"

# 12. delegated commands are printed, not silently dropped
ckc "delegated 100-perm command shown" "NSHUF_HI=100" "$T/l"
ckc "delegated fold-disjoint command shown" "FOLDDISJ_APPLY=1" "$T/l"
ckc "excluded items are costed" "~68 h" "$T/l"

# 13. ONLY= a stage that does not exist -> nothing ran -> failure
mkfixture; run ONLY=9 > "$T/l" 2>&1
ck "an unknown ONLY= exits non-zero" "$?" "1"
ckc "  and says nothing ran" "NOTHING RAN" "$T/l"

echo "--"
[ "$fails" -eq 0 ] && { echo "All checks passed."; exit 0; }
echo "FAILED: $fails check(s)"; exit 1
