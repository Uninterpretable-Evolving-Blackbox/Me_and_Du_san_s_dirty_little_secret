> **Running something for me? Read [START_HERE.md](START_HERE.md).** The other
> `RUN_*.md` files are older and some contradict it.

# Controlled MLM-vs-CLM protein LM experiment

## 👉 START HERE

## 🤖 If you are an agent reading this on Ronnie's box

Do these two things. Both are read-only, need no GPU, and take under a minute.

```bash
bash tests/run_all.sh                      # 126 checks, no GPU, no checkpoints needed
```

Run it before anything else. It confirms the checkout is sound. If it fails, stop and report
the failing suite — do not repair it and continue.

```bash
python verify_paper_claims.py --results <dir with the unpacked batch or its .tgz files>
```

Run it after any batch finishes, **before** reporting a number off it. It recomputes every
figure in the paper the batch covers and prints the paper's value beside the computed one.
Exit 0 means everything matched; exit 1 means at least one claim CHANGED and the report names
which; MISSING means the batch does not cover that claim, which is not a failure.

**The failure mode on this project is plausible numbers, not crashes.** Two aggregation
mistakes have already shipped here — a pivot that paired shuffled seed 42 against real seeds
43/44, and a concept-F1 figure that was a max over depths compared against a per-depth mean.
Both looked right. So do not hand-aggregate CSVs into a comparison; if the comparison you want
is not already in a `RESULT_SUMMARY.md` or in `verify_paper_claims.py`, say so rather than
computing it ad hoc.

**Only one section of this file is an instruction to you: "WHAT TO RUN, IN THIS ORDER".**
Everything after the "Already done" table is either historical or first-time setup for a fresh
machine. This box is already set up. Do not run anything from those sections — several of them
train models for a day or more and their results are already in hand.

Every run script is resumable and skips finished work, so interrupting and re-running is safe.
An empty stage exits non-zero on purpose: it is a failure, not a green tick.

Writing the appendices? [APPENDIX_SOURCES.md](APPENDIX_SOURCES.md) maps each of the twelve cited
appendices to the file its numbers come from, says which are in hand and which are still missing,
and lists the nine claims the delivered data has moved since the current PDF was built.

Both zero-compute asks were delivered in PR #3 on 2026-09-03: the native
`outputs_ctrl_folddisj` cells and all fifteen one-copy scripts. Nothing outstanding there.

**PRs #1, #2 and #3 are all still unmerged**, and hold result archives that exist on no branch.
Fetch them with `git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'`.

---

## ⚠️ FIRST: `git pull` before you run ANYTHING

```bash
cd <this repo> && git pull
```

An audit on 2026-07-25 found a bug in the cleanup step of `run_full_ctrl.sh` and
`run_c2_pertoken.sh`. It checked that *some* results were safely computed, then deleted the big
intermediate files for **everything**, including runs it had never checked. If you ran the
multi-seed job with the old version it would have finished by deleting the intermediate files for
your existing seed-42 results — which cannot be regenerated, because the scripts see the finished
output and skip the step that would recreate them. Fixed: each cleanup now deletes only what it
actually verified, and prints how many files it deliberately kept. Nothing else about how you run
these changed.

---

## 👉 WHAT TO RUN, IN THIS ORDER

**Five runs and one check, about eight hours, none of them training.** Cheapest first, so a long one never
blocks a short one. All are resumable and skip finished work. Full detail in
[START_HERE.md](START_HERE.md), job 6.

```bash
git pull

# does this directory still exist? if not, say so — it decides whether Appendix C is 15 min or 50 GPU-h
ls ~/own_sae_data/uniref50_pilot_shuf_500tpp

STAGE=13 bash RUN_MUSTRUNS.sh                 # ~2 h CPU   continuous separation sweep
STAGE=1  bash run_ctrl_mechanism.sh           # ~2 h CPU   directional contact split
STAGE=10 bash RUN_MUSTRUNS.sh                 # ~1 h GPU   no-autoencoder L_struct, perplexity, raw acts
STAGE=11 bash RUN_MUSTRUNS.sh                 # ~1 h CPU   pairwise probes  <-- MUST follow STAGE=10
NSHUF_HI=25 ONLY=2 bash run_checks.sh         # ~1-2 h CPU larger permutation null
```

Then check the batch before reading anything off it:

```bash
python verify_paper_claims.py --results <dir with the unpacked batch or its .tgz files>
```

**Why this order, and the one dependency that matters.** `STAGE=11` runs its SAE arm happily
without `STAGE=10`, prints one `bad` line about the missing raw activations, and carries on. The
raw arm is the half that answers the reviewer, so run 10 first or you get half the experiment and
a green tick. `STAGE=10` is worth its GPU hour on its own: it also produces the no-autoencoder
L_struct behind §3's "No dictionary at all" check and the perplexities in §2.3, neither of which
has ever been delivered.

**What each closes.** 13 → Appendix F and Figure 2, the only figure that cannot currently be
drawn. 1 → Appendix K and the −0.1422 interaction, which has no source anywhere. 10 → two
unsourced paper numbers plus the input for 11. 11 → the co-activation assumption in §2.2.

**On the last line:** `NSHUF_HI=100` costs ~4–8 h against ~1–2 h at 25, because cost scales with
the permutation count. §2.2's own sensitivity result already bounds the effect — 5 → 25 moves
mean L_struct by +0.0073, never more than +0.0187, and changes no cell's sign — so 25 is very
likely enough. Skip it entirely if time is short; everything above it is worth more.

**Also still outstanding, and free:** nothing. The two zero-compute asks were both delivered in
PR #3 on 2026-09-03 — the native fold-disjoint cells and all fifteen one-copy scripts. Thank you.

### Already done — do not re-run

The 2026-09-02 batch and PR #3 covered these, and
`verify_paper_claims.py` reports 22 pass / 2 changed / 0 missing on it:

| | run | result |
|---|---|---|
| `ONLY=1 bash RUN_TIER1.sh` | top1_share agreement | agrees on both cells; the check count stays at six |
| `ONLY=1 bash RUN_DEPTH_GRID.sh` | corpus control at nine depths | 52/54 cells rise |
| `ONLY=2 bash RUN_TIER1.sh` | fixed / rank denominator | 18/18 rise under all four denominators |
| `ONLY=2 bash RUN_DEPTH_GRID.sh` | probes at block 18 | causal favoured in 3/9, not 27/27 — helix and burial reverse |
| `ONLY=3 bash RUN_TIER1.sh` | d_struct at three seeds | causal 9/9, masked 0/9, at both gates |
| `ONLY=4 bash RUN_TIER1.sh` | d_struct untrained baseline | passes at both gates and both arms |
| `bash RUN_BLOCKSHUFFLE.sh` | second destruction procedure | 14/18 rise — causal 9/9, masked 5/9 |
| `ONLY=5 bash RUN_TIER1.sh` | Benjamini–Hochberg across nine depths | **8/9** survive, not 7/9; plus an omnibus d = +0.2413 [+0.1927, +0.2913] |
| `FOLDDISJ_APPLY=1 ONLY=3 bash run_checks.sh` | fold-disjoint refit, native arm | Δ = +0.01410, matching §2.4 exactly |

---

<details>
<summary><b>Historical: the jobs above superseded these. Both are FINISHED — do not run them.</b></summary>

> ⚠️ Everything inside this block is kept only for the record. The commands here train
> models for 1–2 days and their results are already in hand. If you are deciding what to
> run, the list is at the top of this file, not here.

Two jobs, back to back. Everything else is on hold.

```bash
git pull

# 1. a few hours — analysis only, no training
bash run_c2_pertoken.sh

# 2. ~1-2 days — TRAINS MODELS. The most important run in the project.
PROTOCOLS="token" SEEDS="43 44" bash run_full_ctrl.sh
```

Run **1 first** even though 2 is more important: the GPU does one thing at a time, and 1 finishes
in an evening, so doing it first gets us that answer ~30 hours sooner while delaying 2 by only a
few hours.

**Why 2 matters more than anything else here.** Right now there is exactly one masked model and
one causal model. Every difference we report compares two individual training runs, so we cannot
tell a real effect from ordinary run-to-run variation. Two more seeds per arm turns that into an
actual experiment. `PROTOCOLS="token"` halves the work by running only the headline comparison —
the other protocol was already checked at seed 42.

Both are resumable and skip finished work, so interrupt them freely and just re-run.

**Please don't run these yet:** `run_ctrl_rigor.sh` stages 1/2/4/5 and `run_crosscoder_ctrl.sh`.
They use code written in the last two days that is still being validated on Wei's machine. He'll
tell you when.

**And please keep `~/own_sae_data/`.** Those trained checkpoints are what everything reuses.

### What to expect, and when to ping me

So nothing surprises you — job 2 in particular is much bigger than anything you've run recently.

| | job 1 (`run_c2_pertoken.sh`) | job 2 (`run_full_ctrl.sh`) |
|---|---|---|
| trains models? | **no** — analysis only | **yes**, 4 models |
| wall clock | a few hours | **~1–2 days** |
| GPU busy | intermittently | ~45 min per model, then mostly **CPU** |
| peak disk | ~15 GB, freed at the end | **~56 GB**, freed at the end |

**Two things that look like failure but aren't.**

1. **Long silences with the GPU idle.** Most of the time is the structural-locality metric and
   the confidence intervals, which are CPU-only and single-line-per-cell. Ten to thirty quiet
   minutes is normal. `nvidia-smi` showing 0% does not mean it died — check the log timestamp.
2. **Disk climbing to ~56 GB.** Intermediates are held until the very end because the confidence
   intervals need them, then deleted in one go. If you're tight on space, tell me before starting
   rather than during.

**Please ping me at these three points:**

```bash
# (a) ~1 hour into job 2 — just so we catch a misconfiguration on day 1, not day 2
tail -5 train.log

# (b) when job 1 finishes
tar czf c2_results.tgz $(find outputs_ctrl -name 'struct_seq_metrics.csv' -o -name 'META.json' | grep ctrl_) \
    results_concept_f1 outputs_robustness/bootstrap_h1_c2_*.csv

# (c) when job 2 finishes
tar czf ctrl_seeds.tgz $(find outputs_ctrl -name 'struct_seq_metrics.csv' -o -name 'META.json') \
    results_concept_f1 outputs_robustness/bootstrap_h1_ctrl_esmc_s4*.csv train.log
```

Both tarballs are a few MB — the big files stay on your machine. Each script also prints its own
`Send back` list when it finishes, so if the commands above drift, trust the script's version.

If anything errors, send the last ~50 lines of the log plus:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

---


</details>

<details>
<summary>Older experiments (already done — kept for reference)</summary>

1. [The 15-minute validity check](#one-extra-15-min-run-if-you-still-have-the-checkpoints) ✅
2. [The token ablation](#-next-experiment-the-token-ablation-this-is-the-one-we-need-now) ✅
3. [The C2 instrument fix](#-c2-the-instrument-fix) — this is job 1 above
4. [The crosscoder](#-the-crosscoder-shared-vs-objective-specific-features) — on hold

</details>

Also: **please don't delete `~/own_sae_data/`** — those trained models are what everything
below reuses.

**First time here?** Start at Step 1 below.

---

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

---

# ⭐ NEXT EXPERIMENT: the token ablation (this is the one we need now)

**If you only run one thing, run this.**

```bash
git pull
nohup bash run_token_ablation.sh > token_ablation.log 2>&1 &
tail -f token_ablation.log
```

**What it is.** The models you trained saw 660M tokens ≈ **16 tokens per parameter**. Real
protein language models (ESM-C, ESM-2) are trained at **thousands** of tokens per parameter —
100–1000× more. So when our small models behave differently from the real ones, we can't tell
if that's about model *size* or simply about *how long they trained*. This holds size fixed and
varies training length, which is the cheaper and more informative axis.

**How long.** ~25 h per model, **~50 h total** for both (measured 233k tok/s on your card).
Fully resumable — Ctrl-C, crashes, reboots are all fine, just re-run the same command.

Want a shorter version first? This still gives a 6× span in ~10 h:
```bash
MAX_TOKENS=4.2e9 MILESTONES=0.66e9,2.1e9,4.2e9 bash run_token_ablation.sh
```

**Disk:** ~4 GB (checkpoints along the way are model-only, ~170 MB each).

**What to send back:** just `results_token_ablation/summary.json` (a few KB) and
`token_ablation.log`. Nothing large.

**It won't disturb your earlier run** — everything goes to a separate
`~/own_sae_data/token_ablation/` folder.

---

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

# 🔧 C2: the instrument fix

**Analysis only — no training. A few hours. Run this before the crosscoder.**

```bash
git pull
bash run_c2_pertoken.sh
```

**Why.** The validity check found that the MLM sparse autoencoders are *degenerate* at the
shallowest depths (`val_EV >= 0.99` at blocks 0 and 4 — the same threshold that got an earlier
model dropped from the paper entirely). At that point the autoencoder isn't really compressing
anything, so the structural-locality numbers there can't be trusted. The fix is to normalise
each residue's activation vector to a fixed length before the autoencoder sees it, which
removes the handful of enormous activations that were soaking up all its capacity.

This re-runs extract → normalise → autoencoder → locality + concept alignment + confidence
intervals, for both arms at all 9 depths, and also the un-normalised baseline so we can plot
before/after. Resumable and idempotent — interrupt it freely.

> **Note:** an earlier version of this script deleted `Z.npy` before the confidence intervals
> and the second lens had used it, which would have left us with point estimates and no way to
> recover. Fixed — but if you pulled before 2026-07-25, `git pull` again.

---

# 🧬 The crosscoder: shared vs objective-specific features

**Analysis only — no training. ~2–4 h. This is the newest and most interesting one.**

**This one runs in three stages, and I'd like to look at the output between each.** Not
bureaucracy — the method has a specific failure mode (below) and each stage is a cheap check on
the next one being worth your GPU.

### Stage 0 — one cell, ~15 min. Please do this first and send me the output.

```bash
git pull
STAGE=0 bash run_crosscoder_ctrl.sh
```

Paste me the two lines that look like:

```
Delta_norm median 0.6xx [10-90: 0.6xx, 0.6xx] | absolute bins occupied n/5
Spearman(Delta_norm, struct_delta) = +0.xxxx   <-- PRIMARY
```

**Why I'm asking.** This method can fail in one specific way: if every feature ends up used
*equally* by both models, the comparison has no dynamic range and there is nothing to find. On
my ESM-2/RITA test the spread was narrow — that may just be undertraining, or it may be the
method saying "these two models share everything." Either is a real answer, but I'd rather find
out with 15 minutes of your GPU than 4 hours.

### Stage 1 — seed 42 across depths, ~1–2 h.

```bash
bash run_crosscoder_ctrl.sh
```

Then **stop** and send the results. I'll look before we spend anything on more seeds.

### Stage 2 — the other two seeds, ~2–4 h. Only once I've said go.

```bash
STAGE=2 bash run_crosscoder_ctrl.sh
```

### About the hyperparameters

I may send you three numbers to pass in:

```bash
EXPANSION=8 K_FRAC=0.20 EPOCHS=120 bash run_crosscoder_ctrl.sh
```

These get chosen on the *older* ESM-2-vs-RITA models on my machine, not on yours, and then
frozen. That's deliberate: those older models confound four things at once so they can't support
the main claim anyway, which makes them the right place to do the tuning. Your models differ in
exactly one thing (masked vs causal), so they're where the actual result comes from — and
nothing about how they're analysed gets tuned on the answer. **Please don't tune these.**

**Why this one is different.** Everything we've run so far trains a *separate* feature
dictionary for the masked model and for the causal model, then compares statistics of the two
sets. But the two dictionaries have no correspondence — feature #57 in one has nothing to do
with feature #57 in the other. So the question we actually care about was never even
expressible:

> Are the features that encode 3D structure **shared** between the two training objectives, or
> do they exist **only** in the masked model?

A *crosscoder* trains ONE dictionary that has to reconstruct **both** models at once, with a
separate output matrix per model. Now every feature is a single shared object, and how much
each model relies on it says whether it's shared or specific to one objective. Both answers are
interesting: "unique to masked" means the objective creates a whole family of structural
features; "shared but sharper" means both models build them and bidirectionality just refines
them.

**Why it runs on your box rather than Wei's.** Both arms here are the *same* architecture — one
backbone, one initialisation, only the objective differs. The published diagnostic that checks
whether a "model-specific" feature is real requires both models to have the same width, so it
works here and does **not** work on the older ESM-2-vs-RITA pair (1280 vs 1536). This pair is
the only place the claim can actually be confirmed.

Defaults: 3 depths (25%/50%/75% — the validity check said 0% and 13% aren't trustworthy) × 3
seeds. Skips finished cells, so you can stop and restart.

```bash
ALL=1 bash run_crosscoder_ctrl.sh          # also the full 9-depth profile at seed 42
DEPTHS="14" SEEDS="42" bash run_crosscoder_ctrl.sh   # one quick cell, ~15 min, to check it works
```

Send back everything except `Z.npy` (the script prints the exact `tar` line, ~a few MB).

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
| `run_token_ablation.sh` | rank vs training length |
| `measure_rank_ev.py` / `run_validity_check.sh` | effective rank + SAE val_EV (validity check) |
| `run_c2_pertoken.sh` | the instrument fix — per-token normalisation, redone end to end |
| `crosscoder.py` / `eval_crosscoder_ctrl.py` / `run_crosscoder_ctrl.sh` | **the current experiment** — one shared dictionary across both arms; shared vs objective-specific features |
| `prep_controlled_corpus.py` / `fetch_pdbs.py` | data setup |
| `cache/`, `eval_set/` | precomputed features + the exact 1,500 eval proteins |

`esm==3.2.3` is **pinned deliberately** — `model_ctrl_esmc.py` patches one of its methods
and an upgrade could silently diverge. Please don't `pip install -U esm`.

## If something breaks

Send Wei the last ~50 lines of `train.log` plus:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```
