# Controlled MLM-vs-CLM protein LM experiment

Thanks for lending the GPU! This runs a complete experiment on your machine and sends
back a handful of small CSVs. It's automated — realistically it's **four commands**,
then you leave it alone.

**What it's doing (30 seconds of context).** We're testing whether a language model's
*training objective* changes **where** in the network 3D-protein-structure information
ends up. So we train pairs of models identical in every respect — same architecture,
same data, same batch order, same compute — except one is masked (BERT-style,
bidirectional) and the other causal (GPT-style, left-to-right). Then we measure, layer
by layer, how spatially clustered the learned features are — plus a second independent
check (how well features align with known protein concepts) and bootstrap confidence
intervals. Everything runs here: training, feature extraction, and all the analysis.
Only ~2 MB of CSVs come back.

---

## What you need

- **NVIDIA GPU** (built for your RTX PRO 6000)
- **~100 GB free disk** (1 GB corpus + 1.1 GB structures + checkpoints + ~40 GB of
  intermediates that are auto-deleted at the end, once they've been consumed)
- **Python 3.10+**, internet for setup
- **No bioinformatics tooling.** No DSSP, no BLAST. Precomputed features ship in `cache/`.

---

## 1. Install torch

⚠️ **The RTX PRO 6000 is Blackwell (sm_120) — it needs a CUDA 12.8+ wheel.** Older
`cu121` builds install fine then die at runtime with *"no kernel image is available"*.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify — this must print your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If it says `False`, stop and tell Wei. Nothing else will work.

## 2. Get the data (once)

```bash
python fetch_pdbs.py                  # ~1.1 GB of protein structures (resumable)
python prep_controlled_corpus.py      # ~30-60 min, builds the ~1 GB training corpus
```

`prep` should end with roughly `kept=3000000 seqs | ~1056.7M tokens`.

**A small difference in the token count is expected and fine.** HuggingFace's streaming
shuffle isn't bit-reproducible across `datasets` versions and dataset revisions, so you'll
draw a slightly different 3M sequences (±a few 0.1M tokens is <0.05% of the corpus). It
doesn't matter: both models train on **your** `tokens.npy`, and that shared corpus is what
the experiment controls for. Your run doesn't need to match anyone else's.

Only flag it if one of these is true:
- **`kept` is not 3000000**, or
- the last line shows **`scope-holdout=0`** — that would mean the evaluation proteins
  leaked into training, which *does* invalidate the run. (Ours drops 29.)

## 3. Smoke test (~10 min — please do this first)

```bash
SMOKE=1 bash run_full_ctrl.sh
```

Runs the entire chain on a tiny corpus. It proves your install works before you commit
hours. If it finishes printing `DONE`, you're good.

## 4. Run it

```bash
nohup bash run_full_ctrl.sh > train.log 2>&1 &
tail -f train.log
```

**How long?** We genuinely don't know your card's speed — it was ~12.5 h per model on a
laptop GPU, and we'd *guess* **~1–2 h** on yours. You'll know in 60 seconds, because every
log line prints the live rate and an ETA:

```
step 500/40283 loss 2.9417 lr 6.00e-04 8.2M tok 47s | 174.3k tok/s | ETA 1.0h
```

Stage 1 is **3 training runs + analysis** — plausibly **6–20 h total**. If the ETA looks
insane, or you see `WARNING: no GPU found`, stop and tell Wei.

**Safe to interrupt.** Ctrl-C, crashes, reboots — just re-run the same command. It resumes
from the last checkpoint and skips anything already finished.

## 5. Send back

At the end it prints a checklist. We need the small files only:

```bash
tar czf ctrl_results.tgz \
  $(find outputs_ctrl -name 'struct_seq_metrics.csv' -o -name 'META.json') \
  results_concept_f1 outputs_robustness/bootstrap_h1_ctrl_esmc_*.csv train.log
```

(The script prints this exact command at the end — copy it from there if in doubt.)

That's roughly **2 MB**. Please include `train.log` — the loss curves tell us whether the
two models trained comparably, which the result depends on.

> If the run ends with **"REFUSING to prune Z.npy"**, the bootstrap didn't finish. Nothing
> was deleted — send Wei the log and *don't* clear the directories; the run can resume.

### One extra 15-min run, if you still have the checkpoints

We need one diagnostic the first run didn't record: how many dimensions each model's
activations actually use, and whether the sparse-autoencoder basis is degenerate. It
reads the checkpoints you already have and writes one small JSON:

```bash
bash run_validity_check.sh              # ~15-40 min (rank + SAE val_EV, all depths)
# or, much faster, rank only:
NO_EV=1 bash run_validity_check.sh      # ~5 min, no autoencoder
```

Send back `results_rank_ev/summary.json` (a few KB). This one matters — it decides
whether the main measurement is trustworthy for this pair of models.

### ⛔ Please KEEP the trained checkpoints

**Do not delete `~/own_sae_data/`.** The three `model_final.pt` checkpoints are the
expensive, irreplaceable output — everything else is re-derivable from them in minutes.

Please also send these back (~500 MB each, ~1.5 GB total):

```
~/own_sae_data/uniref50_pilot/ckpt_clm_s42/model_final.pt
~/own_sae_data/uniref50_pilot/ckpt_mlm_s42_token/model_final.pt
~/own_sae_data/uniref50_pilot/ckpt_mlm_s42_pred/model_final.pt
```

`outputs_ctrl/` is safe to delete once the CSVs above are sent.

---

## ⚠️ Please don't tune the hyperparameters

This is the one thing that would silently ruin the experiment. Batch size (32), learning
rate, depth and token budget in `run_full_ctrl.sh` are **pinned to match runs we already
have**. Batch 32 looks absurdly small on a 96 GB card and it is *very* tempting to raise
it — **please don't.** Changing the recipe changes the optimisation, and then the models
aren't comparable to each other or to our existing runs, which defeats the whole point.
Speed should come from the GPU, not a different recipe.

The only things that vary are `--objective`, `--seed`, and the matching protocol. Those
*are* the experiment.

---

## What's actually being run

Two "fair compute" definitions, because they disagree and reviewers ask:

| protocol | tokens seen | predictions made |
|---|---|---|
| **token-matched** | equal (660M each) | CLM makes ~6.7× more |
| **prediction-matched** | MLM gets 6.7× more (4.4B) | equal |

`--match-predictions` only lengthens the **MLM** arm, so the CLM run is shared between
both protocols and trained once. Stage 1 = seed 42 × both protocols = **3 training runs**.

Later, if asked: `SEEDS="43 44" bash run_full_ctrl.sh` adds replicates with no new setup.

## Files

| File | What it is |
|---|---|
| `run_full_ctrl.sh` | **the one you call** — train → extract → SAE → L_struct |
| `model_ctrl_esmc.py` | the 42M model. Uses EvolutionaryScale's own ESM-C stack (`esm==3.2.3`) |
| `train_ctrl_plm.py` | trains one model |
| `eval_ctrl_plm.py` | extracts features + trains the sparse autoencoder |
| `cpu_stage.py` | computes the structural-locality metric (CPU, multi-core) |
| `experiment_concept_f1.py` | second, independent lens: feature<->concept alignment |
| `outputs_robustness/compute_h1_bootstrap.py` | the confidence intervals |
| `measure_rank_ev.py` / `run_validity_check.sh` | effective rank + SAE val_EV (validity check) |
| `prep_controlled_corpus.py` / `fetch_pdbs.py` | data setup |
| `cache/`, `eval_set/` | precomputed features + the exact 1,500 eval proteins |

`esm==3.2.3` is **pinned deliberately** — `model_ctrl_esmc.py` patches one of its methods
and an upgrade could silently diverge. Please don't `pip install -U esm`.

## If something breaks

Send Wei the last ~50 lines of `train.log` plus:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```
