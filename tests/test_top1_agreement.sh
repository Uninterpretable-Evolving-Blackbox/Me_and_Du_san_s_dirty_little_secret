#!/usr/bin/env bash
# Smoke test for analyze_top1_agreement.py. No real data required.
set -u
cd "$(dirname "$0")/.."
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
PY="${PY:-python3}"; fails=0
ck(){ if [ "$2" = "$3" ]; then echo "  [PASS] $1"; else echo "  [FAIL] $1 (got '$2' want '$3')"; fails=$((fails+1)); fi; }

mk(){ # mk <file> <rho_S> <rho_top1>
  printf 'cell,n_features,rho_selectivity_struct,p_struct,rho_top1_struct,p_top1,mean_selectivity\n' > "$1"
  printf 'mlm_s42:11,2560,%s,0.02,%s,0.03,0.31\n' "$2" "$3" >> "$1"
  printf 'clm_s42:11,2560,-0.122,0.001,-0.140,0.001,0.29\n' >> "$1"
}

echo "== analyze_top1_agreement.py =="

mk "$T/agree.csv" -0.044 -0.061
$PY analyze_top1_agreement.py --summary "$T/agree.csv" --per-feature /nonexistent \
   --out "$T/o1.csv" > "$T/l1" 2>&1
ck "agreement case exits 0" "$?" "0"
ck "agreement case says AGREE" "$(grep -c 'VERDICT: AGREE' "$T/l1")" "1"
ck "agreement case keeps six checks" "$(grep -c 'stays at six' "$T/l1")" "1"

mk "$T/sign.csv" -0.044 +0.130
$PY analyze_top1_agreement.py --summary "$T/sign.csv" --per-feature /nonexistent \
   --out "$T/o2.csv" > "$T/l2" 2>&1
ck "opposite-sign case exits 0" "$?" "0"
ck "opposite-sign case says DISAGREE" "$(grep -c 'VERDICT: DISAGREE' "$T/l2")" "1"
ck "opposite-sign case names the reason" "$(grep -c 'opposite sign' "$T/l2")" "1"
ck "opposite-sign case tells you to drop to five" "$(grep -c 'six to five' "$T/l2")" "1"

mk "$T/mag.csv" -0.044 -0.480
$PY analyze_top1_agreement.py --summary "$T/mag.csv" --per-feature /nonexistent \
   --out "$T/o3.csv" > "$T/l3" 2>&1
ck "large-|rho| case says DISAGREE" "$(grep -c 'VERDICT: DISAGREE' "$T/l3")" "1"

# both inside the null band with opposite signs -> sign is not interpretable
mk "$T/null.csv" -0.010 +0.020
$PY analyze_top1_agreement.py --summary "$T/null.csv" --per-feature /nonexistent \
   --out "$T/o4.csv" > "$T/l4" 2>&1
ck "null-band signs do not trigger a false alarm" "$(grep -c 'VERDICT: AGREE' "$T/l4")" "1"

$PY analyze_top1_agreement.py --summary "$T/missing.csv" --out "$T/o5.csv" > "$T/l5" 2>&1
ck "missing input exits 2" "$?" "2"
ck "missing input prints the generating command" "$(grep -c 'experiment_aa_selectivity.py' "$T/l5")" "1"

printf 'cell,something_else\nx,1\n' > "$T/bad.csv"
$PY analyze_top1_agreement.py --summary "$T/bad.csv" --out "$T/o6.csv" > "$T/l6" 2>&1
ck "malformed input exits 2" "$?" "2"

printf 'cell,n_features,rho_selectivity_struct,p_struct,rho_top1_struct,p_top1,mean_selectivity\n' > "$T/empty.csv"
$PY analyze_top1_agreement.py --summary "$T/empty.csv" --out "$T/o7.csv" > "$T/l7" 2>&1
ck "empty input exits 2 (not a silent pass)" "$?" "2"

# per-feature path: direct rho(S, top1) column must appear
mk "$T/pf_sum.csv" -0.044 -0.061
$PY - "$T/pf.csv" <<'PYEOF'
import sys, numpy as np, pandas as pd
rng = np.random.default_rng(0); n = 500
s = rng.random(n); t = 0.8*s + 0.2*rng.random(n)
pd.DataFrame(dict(cell=["mlm_s42:11"]*n, feature=np.arange(n), selectivity=s,
                  top1_share=t, struct_delta=rng.normal(size=n),
                  top_aa=["C"]*n)).to_csv(sys.argv[1], index=False)
PYEOF
$PY analyze_top1_agreement.py --summary "$T/pf_sum.csv" --per-feature "$T/pf.csv" \
   --out "$T/o8.csv" > "$T/l8" 2>&1
ck "per-feature direct correlation is computed" \
   "$($PY -c "import pandas,sys; d=pandas.read_csv('$T/o8.csv'); print(int(d['rho_selectivity_vs_top1'].notna().sum()))")" "1"
ck "output csv has one row per cell" \
   "$($PY -c "import pandas; print(len(pandas.read_csv('$T/o8.csv')))")" "2"

echo "--"
[ "$fails" -eq 0 ] && { echo "All checks passed."; exit 0; }
echo "FAILED: $fails check(s)"; exit 1
