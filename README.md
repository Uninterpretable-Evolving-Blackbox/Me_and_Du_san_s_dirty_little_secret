# Controlled MLM-vs-CLM protein LM runs

Thanks for lending the GPU! This trains six small (35M) protein language models and
sends the checkpoints back. It's fully automated — realistically it's three commands
and then you leave it alone.

**What it's doing (30 seconds of context).** We're testing whether a language model's
*training objective* changes **where** in the network 3D-structure information gets
encoded. So we train pairs of models that are identical in every way — same
architecture, same data, same batch order, same compute — except one is trained with a
masked objective (BERT-style, sees both directions) and the other with a causal one
(GPT-style, left-to-right). We already have one such pair. What we don't know is
whether the effect we saw is real or just luck of the random initialisation, so we need
the same pair trained at **3 different seeds**. That's the 6 models.

---

## What you need

- An **NVIDIA GPU** (this is built for your RTX PRO 6000)
- **~60 GB free disk** (1 GB corpus + ~6 GB of checkpoints per model)
- **Python 3.10+**
- Internet for step 2 (streams the dataset once)

---

## 1. Install torch

⚠️ **The RTX PRO 6000 is Blackwell (sm_120), so it needs a CUDA 12.8+ wheel.** Older
`cu121` builds will install fine and then fail at runtime with "no kernel image is
available". Use:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check it worked — this must print your GPU name:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that says `False`, stop here and tell Wei — nothing else will work.

---

## 2. Build the corpus (once, ~30–60 min)

Streams 3M UniRef50 protein sequences and tokenises them to a ~1 GB file. Only needs
to be done once; all six models train on this exact same corpus.

```bash
python prep_controlled_corpus.py
```

Writes to `~/own_sae_data/uniref50_pilot/`. When it finishes it prints
`kept=3000000 seqs | ~1056.7M tokens` — if your numbers differ, tell Wei before
continuing (it means the corpus doesn't match ours and the comparison breaks).

> Quick 2-minute sanity check first, if you like: `python prep_controlled_corpus.py --smoke`

---

## 3. Train (the long bit)

```bash
bash run_multiseed_cuda.sh
```

That's it. It trains 6 models back-to-back (seeds 42, 43, 44 × masked/causal).

**How long?** Honestly, we don't know your card's rate — it took ~12.5 h per model on a
laptop GPU, and we'd *guess* **1–2 h each** on yours (so ~6–12 h total), but that's an
extrapolation. You'll know within a minute: each log line prints the live rate and an ETA:

```
step 500/40283 loss 2.9417 lr 6.00e-04 8.2M tok 47s | 174.3k tok/s | ETA 1.0h
```

If the ETA looks insane, or it says `WARNING: no GPU found`, stop and tell Wei.

**It's safe to interrupt.** Ctrl-C, crashes, reboots — just re-run the same command. It
resumes from the last checkpoint and skips models that already finished. You can also
run a subset: `SEEDS="42" bash run_multiseed_cuda.sh`.

**Running it in the background** (so an SSH drop doesn't kill it):

```bash
nohup bash run_multiseed_cuda.sh > train.log 2>&1 &
tail -f train.log
```

---

## 4. Send back

When it's done it prints a checklist of the six files. We need:

```
~/own_sae_data/uniref50_pilot/seed{42,43,44}/ckpt_{mlm,clm}/model_final.pt
```

Six files, **380 MB each (~2.3 GB total)** — a shared drive or `rsync` is easiest.
Please include `train.log` too; the loss curves tell us whether the models trained
comparably, which matters for the result.

Once they're sent you can delete `~/own_sae_data/` — nothing else is needed.

---

## ⚠️ Please don't tune the hyperparameters

This is the one thing that would silently ruin the experiment. The batch size (32),
learning rate, and token budget in `run_multiseed_cuda.sh` are pinned to match the runs
we already have. They look small for a 96 GB card and it's very tempting to raise the
batch size for speed — **please don't**. Changing the recipe changes the optimisation,
and then the seeds aren't comparable to each other or to our existing pair, which
defeats the entire point. Speed should come from the GPU, not a different recipe.

The only thing that varies between runs is `--seed`. That's deliberate — it *is* the
experiment.

---

## Files

| File | What it is |
|---|---|
| `prep_controlled_corpus.py` | Streams + tokenises UniRef50 → shared corpus |
| `model_ctrl_plm.py` | The 35M model (one backbone, masked or causal) |
| `train_ctrl_plm.py` | Trains one model |
| `run_multiseed_cuda.sh` | Runs all 6 (this is the one you call) |
| `cache/scope_40.fa` | Protein structures held out of training |

## If something breaks

Send Wei the last ~50 lines of `train.log` plus the output of:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```
