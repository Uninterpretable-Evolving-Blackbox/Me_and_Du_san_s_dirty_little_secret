# Controlled MLM-vs-CLM protein LM experiment

Thanks for lending the GPU! This runs a complete experiment on your machine and sends
back a handful of small CSVs. It's automated — realistically it's **four commands**,
then you leave it alone.

**What it's doing (30 seconds of context).** We're testing whether a language model's
*training objective* changes **where** in the network 3D-protein-structure information
ends up. So we train pairs of models identical in every respect — same architecture,
same data, same batch order, same compute — except one is masked (BERT-style,
bidirectional) and the other causal (GPT-style, left-to-right). Then we measure, layer
by layer, how spatially clustered the learned features are. Everything runs here:
training, feature extraction, and the analysis. Only ~2 MB of CSVs come back.

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

`prep` should end with `kept=3000000 seqs | ~1056.7M tokens`. **If your numbers differ,
tell Wei before continuing** — it means the corpus doesn't match ours.

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
  outputs_robustness/bootstrap_h1_ctrl_esmc_*.csv train.log
```

That's roughly **2 MB**. Please include `train.log` — the loss curves tell us whether the
two models trained comparably, which the result depends on.

> If the run ends with **"REFUSING to prune Z.npy"**, the bootstrap didn't finish. Nothing
> was deleted — send Wei the log and *don't* clear the directories; the run can resume.

Then delete `~/own_sae_data/` and `outputs_ctrl/` — nothing else is needed.

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
| `prep_controlled_corpus.py` / `fetch_pdbs.py` | data setup |
| `cache/`, `eval_set/` | precomputed features + the exact 1,500 eval proteins |

`esm==3.2.3` is **pinned deliberately** — `model_ctrl_esmc.py` patches one of its methods
and an upgrade could silently diverge. Please don't `pip install -U esm`.

## If something breaks

Send Wei the last ~50 lines of `train.log` plus:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```
