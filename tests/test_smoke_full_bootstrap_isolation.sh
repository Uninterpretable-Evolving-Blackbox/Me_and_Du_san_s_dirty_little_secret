#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cp "$ROOT/run_full_ctrl.sh" "$TMP/run_full_ctrl.sh"
mkdir -p "$TMP/cache/pdb_files" "$TMP/eval_set" "$TMP/outputs_robustness"
mkdir -p "$TMP/home/own_sae_data/uniref50_smoke"
mkdir -p "$TMP/home/own_sae_data/uniref50_pilot"
touch "$TMP/home/own_sae_data/uniref50_smoke/tokens.npy"
touch "$TMP/home/own_sae_data/uniref50_pilot/tokens.npy"

export BOOT_CALLS="$TMP/bootstrap_calls.txt"
: > "$BOOT_CALLS"

cat > "$TMP/python_stub" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail

cmd=${1:-}
if [ "$cmd" = "-" ]; then
  cat >/dev/null
  exit 0
fi
shift
if [ "$cmd" = "-u" ]; then
  cmd=${1:-}
  shift
fi

case "$cmd" in
  train_ctrl_plm.py)
    out=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --out-dir) out=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$out"
    touch "$out/model_final.pt"
    ;;
  eval_ctrl_plm.py)
    name=""; layer=""; root=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --name) name=$2; shift 2 ;;
        --layer) layer=$2; shift 2 ;;
        --out-root) root=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$root/$name/layer_$layer"
    touch "$root/$name/layer_$layer/Z.npy"
    ;;
  cpu_stage.py)
    layer_dir=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --layer-dir) layer_dir=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    touch "$layer_dir/struct_seq_metrics.csv"
    ;;
  experiment_concept_f1.py)
    save_dir=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --save-dir) save_dir=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$save_dir"
    touch "$save_dir/concept_f1.csv"
    ;;
  outputs_robustness/compute_h1_bootstrap.py)
    stem=""; depths=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --out-stem) stem=$2; shift 2 ;;
        --depths) depths=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    printf '%s %s\n' "$stem" "$depths" >> "$BOOT_CALLS"
    if [ "$depths" = "all" ]; then
      labels="0 13 25 38 50 63 75 88 100"
    else
      labels=$(printf '%s' "$depths" | tr ',' ' ')
    fi
    for split in full val; do
      out="outputs_robustness/${stem}_${split}_bylevel_minact0.csv"
      printf 'rel_depth,cluster_level\n' > "$out"
      for label in $labels; do
        printf '%s%%,fold\n%s%%,protein\n' "$label" "$label" >> "$out"
      done
    done
    touch "outputs_robustness/${stem}_traces_bylevel_minact0.npz"
    ;;
  *)
    echo "unexpected stub command: $cmd" >&2
    exit 2
    ;;
esac
STUB
chmod +x "$TMP/python_stub"

(cd "$TMP" && HOME="$TMP/home" PY="$TMP/python_stub" SMOKE=1 bash run_full_ctrl.sh >/dev/null)
(cd "$TMP" && HOME="$TMP/home" PY="$TMP/python_stub" bash run_full_ctrl.sh >/dev/null)

token_calls=$(grep -c 'token' "$BOOT_CALLS" || true)
if [ "$token_calls" -ne 2 ]; then
  echo "expected separate smoke and full token bootstraps; got $token_calls" >&2
  cat "$BOOT_CALLS" >&2
  exit 1
fi

token_stems=$(grep 'token' "$BOOT_CALLS" | awk '{print $1}' | sort -u | wc -l | tr -d ' ')
if [ "$token_stems" -ne 2 ]; then
  echo "expected distinct smoke and full bootstrap stems" >&2
  cat "$BOOT_CALLS" >&2
  exit 1
fi

echo "PASS: smoke and full token bootstrap outputs are isolated"
