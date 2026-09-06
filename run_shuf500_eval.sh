#!/usr/bin/env bash
# run_shuf500_eval.sh — L_struct for the shuffled pair trained to 500 tok/param.
#
# THE QUESTION this closes. On real data the MLM-CLM gap doubles at 500 tok/param
# (+0.0161 -> +0.0306 at L11), so "the models were undertrained" does not explain the
# effect. But the validity failure -- shuffled residues INFLATING L_struct rather than
# collapsing it -- has only ever been measured at 15.7 tok/param. Three seeds at that
# budget now give 18/18 cells above 1.0 (causal 10-70x, masked 1.3-2.1x), but all at one
# training length. If the inflation survives 32x more training the failure is a property
# of the metric; if it vanishes it is a property of undertrained models, which would be a
# more important finding and must be reported either way.
#
# Same recipe as every other cell: sae_seed 42, expansion 8, k 256, cpu_stage with
# --n-shuffles 5 (the project default that 3 would silently change).
#
# Serial, one process at a time.
set -u
cd "$(dirname "$0")"
PY="${PY:-$PWD/.venv/bin/python}"
SHUF500="$HOME/own_sae_data/uniref50_pilot_shuf_500tpp"
OUT=outputs_ctrl_shuf_500tpp
LAYERS="11 14 18"

echo "########## shuffled @ 500 tok/param | $(date) ##########"
ok=0; fail=0
for arm in ckpt_mlm_s42_token ckpt_clm_s42; do
  ck="$SHUF500/$arm/model_final.pt"
  if [ ! -f "$ck" ]; then echo "!! missing $ck"; fail=$((fail+1)); continue; fi
  for L in $LAYERS; do
    LD="$OUT/$arm/layer_$L"
    if [ -f "$LD/struct_seq_metrics.csv" ]; then echo "[$arm L$L] done - skip"; ok=$((ok+1)); continue; fi
    echo "=== [$arm L$L] extract + SAE $(date +%T) ==="
    if ! "$PY" -u eval_ctrl_plm.py --ckpt "$ck" --name "$arm" --layer "$L" \
         --out-root "$OUT" --eval-set eval_set \
         --sae-seed 42 --expansion 8 --k-sparse 256; then
      echo "!! eval failed $arm L$L"; fail=$((fail+1)); continue
    fi
    echo "=== [$arm L$L] cpu_stage $(date +%T) ==="
    if "$PY" -u cpu_stage.py --layer-dir "$LD" --model-type residue --n-shuffles 5 \
         --features-csv cache/residue_features.csv --pdb-dir cache/pdb_files \
         --fasta-path cache/scope_40.fa; then
      ok=$((ok+1))
    else
      echo "!! cpu_stage failed $arm L$L"; fail=$((fail+1))
    fi
  done
done

echo
echo "########## DONE $(date)  ok=$ok fail=$fail ##########"
"$PY" - <<'PY'
import csv, os
import numpy as np

def ms(root, arm, L):
    f = "%s/%s/layer_%d/struct_seq_metrics.csv" % (root, arm, L)
    if not os.path.exists(f):
        return None
    return float(np.mean([float(r["struct_delta"]) for r in csv.DictReader(open(f))]))

print()
print("  shuffled / real ratio, 15.7 vs 500 tokens per parameter (seed 42)")
print()
print("  arm      layer |   15.7: real    shuf   ratio |   500: real    shuf   ratio")
for obj, sfx, lab in (("clm", "", "causal"), ("mlm", "_token", "masked")):
    arm = "ckpt_%s_s42%s" % (obj, sfx)
    for L in (11, 14, 18):
        a = ms("outputs_ctrl", arm, L)
        b = ms("outputs_ctrl_shuf", arm, L)
        c = ms("outputs_ctrl_500tpp", arm, L)
        d = ms("outputs_ctrl_shuf_500tpp", arm, L)
        if None in (a, b):
            continue
        lo = "%+8.5f %+8.5f %7.2fx" % (a, b, b / a)
        hi = ("%+8.5f %+8.5f %7.2fx" % (c, d, d / c)) if None not in (c, d) else "   (incomplete)"
        print("  %-7s  %3d  | %s | %s" % (lab, L, lo, hi))
print()
print("  ratio > 1 at 500 tok/param means the inflation is a property of the metric,")
print("  not of undertrained models.")
PY
