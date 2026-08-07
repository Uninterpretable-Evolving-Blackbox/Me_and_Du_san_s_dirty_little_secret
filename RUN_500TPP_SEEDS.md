# The 500 tok/param comparison is n=1, and it is now a headline

**Ronnie — this is the one expensive ask. Read the priority note at the bottom before
starting it; the InterPLM attack is cheaper and should go first.**

## Why

Your 2026-08-07 budget table is the most informative result in that batch. It showed that
across 32× more training the *shuffled* value barely moves (−4% to +23%) while the real
values move a great deal **and in opposite directions**: causal falls 25–66%, masked rises
13–116%. That is what turns "the metric is broken" into "here is the regime in which this
class of metric is invalid", which is the paper's spine.

Every cell in it is **seed 42**.

The 15.7 tok/param comparison went to three seeds and the masked effect came down 6–14%
when it did — seed 42 was the most favourable of the three at both L11 and L14. So we know
this design has seed-to-seed spread large enough to matter, and we are currently resting a
headline on one draw of it.

There is a second reason. Your table's most quotable cell is masked L14 at 0.90× — shuffled
scoring *below* real, the direction a working metric should show. That single cell is doing
a lot of work for the "roughly valid once well trained" reading, and the other two masked
cells at that budget are 1.32× and 1.75×. One seed cannot tell us whether 0.90 is the
signal or the tail.

## What to train

Four models per seed — the real pair and the shuffled pair, both at 21e9 tokens. The
recipe below is copied from `run_queue_0804.sh` job C and **must not be varied**; the
seed-42 models it has to be compared against were trained at exactly these settings
(`ckpt_tokens = 20,999,995,392` in every `META.json`).

```bash
TARGET=21e9; BATCH=32; SEQ=512; LR=6e-4; WARMUP=500
CKPT_EVERY=5000; VAL_EVERY=2000
D_MODEL=320; N_HEADS=5; N_LAYERS=30

REAL=$HOME/own_sae_data/uniref50_pilot                  # data
REAL_OUT=$HOME/own_sae_data/uniref50_pilot/token_ablation
SHUF=$HOME/own_sae_data/uniref50_pilot_shuf             # data
SHUF_OUT=$HOME/own_sae_data/uniref50_pilot_shuf_500tpp

for s in 43; do                                          # then 44 if there is time
  for obj in clm mlm; do
    case "$obj" in clm) name="ckpt_clm_s${s}";; mlm) name="ckpt_mlm_s${s}_token";; esac
    for pair in "$REAL:$REAL_OUT" "$SHUF:$SHUF_OUT"; do
      DATA="${pair%%:*}"; OUT="${pair##*:}/$name"
      [ -f "$OUT/model_final.pt" ] && { echo "[$OUT] done - skip"; continue; }
      mkdir -p "$OUT"
      res=""; [ -f "$OUT/model_resume.pt" ] && res="--resume $OUT/model_resume.pt"
      "$PY" -u train_ctrl_plm.py --objective "$obj" --data-dir "$DATA" --out-dir "$OUT" \
        --seed "$s" --target-tokens "$TARGET" --batch-size "$BATCH" --seq-len "$SEQ" \
        --lr "$LR" --warmup "$WARMUP" --ckpt-every "$CKPT_EVERY" --val-every "$VAL_EVERY" \
        --rolling-resume --d-model "$D_MODEL" --n-heads "$N_HEADS" --n-layers "$N_LAYERS" $res
    done
  done
done
```

**Serial, one process at a time**, for the reason in `run_queue_0804.sh:8-11`.

## What to evaluate

Training produces **no numbers**. This is the stage-5/stage-16 trap again: without the
evaluation pass you get checkpoints and nothing else.

`run_shuf500_eval.sh` already does exactly the right thing for the shuffled tree — reuse it
with `SHUF500` and `OUT` repointed per seed, and run the same two steps for the real tree:

```bash
"$PY" -u eval_ctrl_plm.py --ckpt "$ck" --name "$arm" --layer "$L" \
      --out-root "$OUT" --eval-set eval_set --sae-seed 42 --expansion 8 --k-sparse 256
"$PY" -u cpu_stage.py --layer-dir "$OUT/$arm/layer_$L" --model-type residue \
      --n-shuffles 5 --features-csv cache/residue_features.csv \
      --pdb-dir cache/pdb_files --fasta-path cache/scope_40.fa
```

Layers 11, 14, 18. `--sae-seed 42 --expansion 8 --k-sparse 256` and `--n-shuffles 5` are
the project defaults every other cell used; `--n-shuffles 3` would silently change the
comparison.

## Send back

`outputs_ctrl_500tpp/` and `outputs_ctrl_shuf_500tpp/` for the new seeds — the
`struct_seq_metrics.csv` **and** the `META.json` beside it. The META is not optional: I read
`val_EV` out of it, and in all 24 existing matched cells the shuffled arm's SAE reconstructs
better than the real arm's, which is a confound I have to be able to quantify per cell.

## Cost, honestly

~50 GPU-hours per pair, so **~100 GPU-h for seed 43** (real pair + shuffled pair) and
~200 GPU-h for both seeds. That is 4 days and 8 days respectively of a box doing nothing
else.

If that is too much, the ranked fallback:

1. **Shuffled pair only, seed 43** (~50 GPU-h). Tests whether the floor's
   training-invariance is seed-stable, which is the load-bearing half.
2. **Masked arm only, both trees, seed 43** (~50 GPU-h). Directly attacks the 0.90× cell,
   which is the number most likely to be challenged.
3. Nothing, and we report n=1 with the limitation stated. Not fatal — but it is the same
   limitation the 15.7 tok/param table just spent three seeds removing.

## Priority

**Run `RUN_INTERPLM_ATTACK.md` first.** It is ~2–3 h of setup plus a grid of SAE fits — no
pretraining — and it closes a gap nothing else can: every result we have used *our*
implementation of InterPLM's metric, never their code. Until that runs we cannot write
"we ran InterPLM's published metric on an objective-isolated pair", only "we ran our
implementation of it". That sentence is worth more than a fourth significant figure on a
seed count.

Then `RUN_NOMODEL_BASELINE.md` (cheap, no GPU). Then this.
