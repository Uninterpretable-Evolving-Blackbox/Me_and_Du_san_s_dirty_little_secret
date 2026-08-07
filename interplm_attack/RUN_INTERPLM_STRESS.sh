#!/bin/bash
# RUN_INTERPLM_STRESS.sh — InterPLM's published concept-F1, run on our models.
# For Ronnie's box. Everything is their code except the backbone.
#
# ---------------------------------------------------------------------------
# WHAT THIS ESTABLISHES, AND WHY IT NEEDS YOUR BOX
# ---------------------------------------------------------------------------
# The laptop can only run the 33.2M pair (n=1 per arm, no shuffled corpus) and
# shut itself down twice under sustained load. You have what the questions need:
# the 42.0M ESM-C pair at 3 seeds per arm, the shuffled-corpus arm, and the
# 500 tok/param pair.
#
# ---------------------------------------------------------------------------
# BEFORE RUNNING — three things that will silently ruin the output
# ---------------------------------------------------------------------------
# 1. LENGTH FILTER. extract_annotations never truncates; our models cap at
#    cfg["max_seq"]. Any protein longer than max_seq-2 trips the alignment
#    guard. STAGE 1 applies the filter. Do not skip it.
# 2. metadata.yaml. Their SAE dataloader silently rejects shards without it and
#    reports four causes, none of them the real one. The adapter writes it.
# 3. THREE ENVIRONMENT PINS, or their code will not run at all:
#       pandas<3          (np.array_split on a DataFrame changed behaviour)
#       nnsight==0.5.15   (their fidelity call is rejected by 0.4.11-0.7.0;
#                          we bypass fidelity, but the import must resolve)
#       SSL_CERT_FILE     (torch.hub cert verification)
#
# ---------------------------------------------------------------------------
# USAGE
#   export SAE_SRC=/path/to/sae_review/src        # model_ctrl_esmc.py lives here
#   export CKPT_ROOT=/home/ronnie/own_sae_data/uniref50_pilot
#   ./RUN_INTERPLM_STRESS.sh setup      # env + downloads + annotations
#   ./RUN_INTERPLM_STRESS.sh smoke      # 2 min: verify the esmc adapter path
#   ./RUN_INTERPLM_STRESS.sh grid       # the full run
# ---------------------------------------------------------------------------
set -eu
BASE=${BASE:-$HOME/interplm_stress}
SAE_SRC=${SAE_SRC:-$(cd "$(dirname "$0")/.." && pwd)}   # model_ctrl_esmc.py is at repo root
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the dir holding ckpt_* checkpoints}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$BASE/venv/bin/python
REPO=$BASE/interplm_repo
DEV=${DEV:-cuda}
LAYERS=${LAYERS:-"11 14 18"}
SEEDS=${SEEDS:-"0 1 2"}
L1=${L1:-0.06}
MAXLEN=${MAXLEN:-510}          # <= max_seq-2 of the models being embedded

# Which checkpoints to run.  name:ABSOLUTE_CKPT_PATH:extra-flags
#
# !! ABSOLUTE PATHS, DELIBERATELY. An earlier version of this file listed the
# !! shuffled arms as bare subdirectories under CKPT_ROOT, which resolved to the
# !! REAL checkpoints. The shuffled-corpus control would then have compared the
# !! real models against themselves and reported "shuffled == real" — a null that
# !! looks like a passing validity check and is entirely an artefact. The
# !! shuffled models live in a SEPARATE data tree (prep_controlled_corpus.py
# !! --shuffle-residues writes to its own out-dir), so they cannot share CKPT_ROOT.
# !!
# !! Set CKPT_ROOT_SHUF to that tree. The script refuses to run the shuffled arms
# !! if it is unset, rather than silently substituting the real models.
CKPT_ROOT_SHUF=${CKPT_ROOT_SHUF:-}

MODELS=${MODELS:-"
mlm_s42:$CKPT_ROOT/ckpt_mlm_s42_token/model_final.pt:
mlm_s43:$CKPT_ROOT/ckpt_mlm_s43_token/model_final.pt:
mlm_s44:$CKPT_ROOT/ckpt_mlm_s44_token/model_final.pt:
clm_s42:$CKPT_ROOT/ckpt_clm_s42/model_final.pt:
clm_s43:$CKPT_ROOT/ckpt_clm_s43/model_final.pt:
clm_s44:$CKPT_ROOT/ckpt_clm_s44/model_final.pt:
shuf_mlm_s42:$CKPT_ROOT_SHUF/ckpt_mlm_s42_token/model_final.pt:
shuf_clm_s42:$CKPT_ROOT_SHUF/ckpt_clm_s42/model_final.pt:
untrained:$CKPT_ROOT/ckpt_mlm_s42_token/model_final.pt:--random-init
"}

# NOTE: the shuffled-arm guard lives inside grid(), NOT here. At top level it
# fired during `setup`, which never touches CKPT_ROOT_SHUF -- Ronnie's very first
# command would exit 1 on an unused variable.

setup () {
  mkdir -p "$BASE"; cd "$BASE"
  [ -d interplm_repo ] || git clone --depth 1 https://github.com/ElanaPearl/interPLM.git interplm_repo
  [ -d venv ] || python3.12 -m venv venv
  $BASE/venv/bin/pip install -q -r interplm_repo/requirements.txt \
      "pandas>=2.2,<3" "nnsight==0.5.15"
  $BASE/venv/bin/pip install -q -e interplm_repo
  export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")

  # Swiss-Prot: annotations (concept labels) + sequences (SAE training data)
  [ -f proteins_full.tsv.gz ] || curl -s -o proteins_full.tsv.gz \
"https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cprotein_name%2Clength%2Csequence%2Cec%2Cft_act_site%2Cft_binding%2Ccc_cofactor%2Cft_disulfid%2Cft_carbohyd%2Cft_lipid%2Cft_mod_res%2Cft_signal%2Cft_transit%2Cft_helix%2Cft_turn%2Cft_strand%2Cft_coiled%2Ccc_domain%2Cft_compbias%2Cft_domain%2Cft_motif%2Cft_region%2Cft_zn_fing%2Cxref_alphafolddb&format=tsv&query=%28reviewed%3Atrue%29"
  [ -f uniprot_sprot.fasta.gz ] || curl -s -o uniprot_sprot.fasta.gz \
"https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"

  # STAGE 1: length filter + 50k subsample. STREAMED -- reading the whole TSV at
  # once is ~10 GB resident and is what wedged the laptop.
  $PY - <<EOF
import pandas as pd, numpy as np
keep=[]
for ch in pd.read_csv("proteins_full.tsv.gz",sep="\t",chunksize=20000,low_memory=False):
    keep.append(ch[ch["Length"]<=$MAXLEN])
df=pd.concat(keep,ignore_index=True)
rng=np.random.RandomState(42)
sub=df.iloc[rng.choice(len(df),size=min(50000,len(df)),replace=False)]
sub.to_csv("proteins_50k.tsv.gz",sep="\t",index=False,compression="gzip")
print(f"len<={$MAXLEN}: {len(df):,} | sampled {len(sub):,} | residues {int(sub.Length.sum()):,}")
EOF

  cd "$REPO"
  $PY -m interplm.analysis.concepts.extract_annotations \
      --input_uniprot_path "$BASE/proteins_50k.tsv.gz" \
      --output_dir "$BASE/ann/processed" --n_shards 16 --max_workers 1 \
      --min_required_instances 10
  $PY -m interplm.analysis.concepts.prepare_eval_set \
      --valid_shard_range 0 7 --test_shard_range 8 15 \
      --uniprot_dir "$BASE/ann/processed" \
      --min_aa_per_concept 1500 --min_domains_per_concept 25

  # SAE training sequences (disjoint from the annotated set by construction)
  $PY scripts/subset_fasta.py --input_file "$BASE/uniprot_sprot.fasta.gz" \
      --output_file "$BASE/train_subset.fasta" --num_proteins 20000 --max_length $MAXLEN
  $PY scripts/shard_fasta.py --input_file "$BASE/train_subset.fasta" \
      --output_dir "$BASE/train_shards/" --proteins_per_shard 4000

  # parameterised trainer: their example, 8 lines changed (paths, dims, L1, seed,
  # and EvaluationConfig in place of ESMFidelityConfig since nnsight is broken)
  sed -e 's|from interplm.train.fidelity import ESMFidelityConfig|from interplm.train.evaluation import EvaluationConfig|' \
      -e 's|    eval_cfg = ESMFidelityConfig(|    eval_cfg = EvaluationConfig(|' \
      -e '/        model_name="esm2_t6_8M_UR50D",/d' -e '/        layer_idx=int(LAYER),/d' \
      -e 's|    EMBEDDINGS_DIR = .*|    EMBEDDINGS_DIR = Path(os.environ["CTRL_EMBD_DIR"])|' \
      -e 's|    SAVE_DIR = .*walkthrough_model.*|    SAVE_DIR = Path(os.environ["CTRL_SAVE_DIR"])|' \
      -e 's|    EMBEDDING_DIM = 320.*|    EMBEDDING_DIM = int(os.environ["CTRL_DIM"])|' \
      -e 's|    HIDDEN_SIZE = 1280.*|    HIDDEN_SIZE = int(os.environ["CTRL_DIM"])*int(os.environ.get("CTRL_EXP","4"))|' \
      -e 's|    L1_COEFFICIENT = 0.06.*|    L1_COEFFICIENT = float(os.environ.get("CTRL_L1","0.06"))|' \
      -e 's|        batch_size=BATCH_SIZE,|        batch_size=BATCH_SIZE,\n        seed=int(os.environ.get("CTRL_SEED","0")),|' \
      examples/train_basic_sae.py > examples/train_ctrl_sae.py

  # B3: the seed MUST go through DataloaderConfig(seed=...), not a manual_seed at
  # the top of main(). InterPLM's ActivationsDataLoader.__init__ and
  # ShardedActivationsDataset.__init__ both call torch.manual_seed(config.seed),
  # and the dataloader is constructed BEFORE the SAE (training_run.py:174 vs 211),
  # so an injected seed is overwritten before a single dictionary weight is drawn.
  # That is why three "seeds" previously produced bit-identical dictionaries.
  #
  # Caveat to state in the manuscript: DataloaderConfig.seed controls dictionary
  # init AND shard/batch order jointly. There is no init-only knob without editing
  # their library, which the one rule forbids. So: "SAE replicates vary dictionary
  # initialisation and data order jointly."
  grep -q 'seed=int(os.environ' examples/train_ctrl_sae.py || {
    echo "ERROR: the CTRL_SEED sed did not match — train_ctrl_sae.py would run"
    echo "       every seed identically. Check batch_size=BATCH_SIZE in their"
    echo "       examples/train_basic_sae.py."
    exit 1
  }
  for pat in CTRL_EMBD_DIR CTRL_SAVE_DIR CTRL_DIM CTRL_L1 EvaluationConfig; do
    grep -q "$pat" examples/train_ctrl_sae.py || {
      echo "ERROR: sed pattern '$pat' did not match — the generated trainer would"
      echo "       silently run with THEIR hyperparameters, not ours."; exit 1; }
  done
  echo "SETUP COMPLETE (generated trainer verified)"
}

smoke () {
  $PY "$HERE/embed_ctrl_interplm.py" --ckpt "$CKPT_ROOT/ckpt_mlm_s42_token/model_final.pt" \
      --arch esmc --src-dir "${SAE_SRC:-$(cd "$HERE/.." && pwd)}" --layer 14 --device "$DEV" --smoke
}

grid () {
  export INTERPLM_DATA=$BASE
  export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
  cd "$REPO"; mkdir -p "$BASE/results"

  # ---- PREFLIGHT (B1) ----------------------------------------------------
  # A non-empty CKPT_ROOT_SHUF is NOT sufficient: setting it equal to CKPT_ROOT
  # passes an emptiness check and silently points the shuffled arms at the REAL
  # checkpoints. The control then compares the real models against themselves
  # and reports "shuffled == real" -- a null that looks like a PASSING validity
  # check and is pure artefact. That control is the paper's central claim, so
  # this is content-addressed: it also catches copies and symlinks.
  if echo "$MODELS" | grep -q "^shuf_" && [ -z "$CKPT_ROOT_SHUF" ]; then
    echo "ERROR: MODELS includes shuffled arms but CKPT_ROOT_SHUF is unset."
    echo "       Shuffled checkpoints live in their own tree (prep_controlled_corpus.py"
    echo "       --shuffle-residues writes a separate out-dir). Set it, or drop shuf_*."
    exit 1
  fi
  SHA=$(command -v sha256sum || echo "shasum -a 256")
  sums=$(echo "$MODELS" | while IFS=: read -r NAME CK EXTRA; do
           if [ -n "${NAME:-}" ] && [ "${EXTRA:-}" != "--random-init" ]; then
             if [ ! -f "$CK" ]; then echo "MISSING $NAME $CK" >&2; exit 1; fi
             echo "$($SHA "$CK" | cut -d' ' -f1) $NAME"
           fi
         done) || { echo "ERROR: a listed checkpoint is missing (see above)"; exit 1; }
  dup=$(echo "$sums" | sort | awk '{if($1==p) print pn" == "$2; p=$1; pn=$2}')
  if [ -n "$dup" ]; then
    echo "ERROR: distinct arms resolve to IDENTICAL checkpoint weights:"
    echo "$dup"
    echo "       The shuffled control would compare a model against itself."
    exit 1
  fi
  echo "preflight OK: $(echo "$sums" | wc -l | tr -d ' ') checkpoints, all distinct"
  for LAYER in $LAYERS; do
    echo "$MODELS" | while IFS=: read -r NAME CK EXTRA; do
      [ -z "$NAME" ] && continue
      if [ ! -f "$CK" ]; then
        echo "ERROR: checkpoint for '$NAME' not found: $CK — refusing to continue."
        exit 1
      fi
      A=$BASE/embd_analysis/$NAME/L$LAYER; T=$BASE/embd_train/$NAME/L$LAYER
      if [ "${EXTRA:-}" = "--random-init" ]; then EXTRA="--random-init --init-seed 42"; fi
      DIM=$($PY -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['cfg']['d_model'])")
      [ -f "$A/.done" ] || $PY "$HERE/embed_ctrl_interplm.py" --ckpt "$CK" --arch esmc \
          --src-dir "${SAE_SRC:-$(cd "$HERE/.." && pwd)}" --layer "$LAYER" --device "$DEV" --model-tag "$NAME" \
          $EXTRA --shards-dir "$BASE/ann/processed" --out "$A"
      [ -f "$T/.done" ] || $PY "$HERE/embed_ctrl_interplm.py" --ckpt "$CK" --arch esmc \
          --src-dir "${SAE_SRC:-$(cd "$HERE/.." && pwd)}" --layer "$LAYER" --device "$DEV" --model-tag "$NAME" \
          $EXTRA --fasta-dir "$BASE/train_shards" --out "$T"

      for S in $SEEDS; do
        TAG=${NAME}_L${LAYER}_s${S}; SAVE=models/grid/$TAG
        CTRL_DIM=$DIM CTRL_L1=$L1 CTRL_SEED=$S CTRL_EMBD_DIR=$T \
        CTRL_SAVE_DIR=$SAVE LAYER=$LAYER $PY examples/train_ctrl_sae.py \
            > "$BASE/results/train_$TAG.log" 2>&1
        # outcome-blind quality FIRST, always, before any concept number exists
        $PY "$HERE/sae_quality.py" --sae "$SAVE/ae.pt" --embd "$A" --repo "$REPO" \
            --tag "$TAG" >> "$BASE/results/sae_quality.txt" 2>&1
        $PY -m interplm.sae.normalize --sae_dir "$SAVE" --aa_embds_dir "$A" >/dev/null 2>&1
        for SP in valid test; do
          $PY -m interplm.analysis.concepts.compare_activations --sae_dir "$SAVE" \
              --aa_embds_dir "$A" --eval_set_dir "$BASE/ann/processed/$SP/" \
              --output_dir "results/grid/${TAG}_${SP}" >/dev/null 2>&1
          $PY -m interplm.analysis.concepts.calculate_f1 \
              --eval_res_dir "results/grid/${TAG}_${SP}" \
              --eval_set_dir "$BASE/ann/processed/$SP/" >/dev/null 2>&1
        done
        { echo "----- $TAG -----"
          $PY -m interplm.analysis.concepts.report_metrics \
              --valid_path "results/grid/${TAG}_valid/concept_f1_scores.csv" \
              --test_path  "results/grid/${TAG}_test/concept_f1_scores.csv" 2>&1 \
            | grep -iE "Average best F1|concepts identified|features associated"
        } >> "$BASE/results/concept_f1.txt"
        $PY "$HERE/aa_floor.py" --valid "$BASE/ann/processed/valid" \
            --test "$BASE/ann/processed/test" --repo "$REPO" \
            --sae-pairings "results/grid/${TAG}_test/heldout_top_pairings.csv" \
            --out "$BASE/results/floor_vs_sae_$TAG.csv" >/dev/null 2>&1
        echo "done $TAG"
      done
      rm -rf "$A" "$T"          # B7: per-model cleanup, peak ~24 GB not ~210 GB
    done
  done
  echo "GRID COMPLETE -> $BASE/results/"
}

case "${1:-}" in
  setup) setup ;; smoke) smoke ;; grid) grid ;;
  *) echo "usage: $0 {setup|smoke|grid}"; exit 1 ;;
esac
