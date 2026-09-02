# Start here

## Run the preflight first

```bash
git pull
./preflight.sh all
```

Every check prints `PASS` / `FAIL` / `SKIP` with the expected value stated, and the script
exits non-zero if anything failed. It takes a couple of minutes and needs no GPU.

**On any `FAIL`: stop and report it. Do not repair and continue.** Every check exists
because the thing it guards has already failed silently at least once here, and the
failure mode in this project is *plausible numbers, not crashes* — which is far more
expensive to catch later than to catch now.

Two more checkpoints during the InterPLM attack, both cheap:

```bash
./preflight.sh post-setup    # after `RUN_INTERPLM_STRESS.sh setup`, before `grid`
./preflight.sh first-cell    # after the first grid cell, before leaving it running
```

`first-cell` is the important one. It verifies that the three SAE seeds produce **distinct
dictionaries** — that exact bug already produced three bit-identical "seeds" in this
project's own results, which would silently make every three-seed number an n=1 number.

---

Ignore the other RUN_*.md files. They are older and some of them contradict this one.

**Four exceptions, all new (2026-08-07) and not superseded by anything here. Run them in
this order — cheapest first, so a red one never blocks a cheap one:**

| | brief | cost | status |
|---|---|---|---|
| 1 | **`RUN_NOMODEL_BASELINE.md`** — builds one layer dir whose features contain no model, which any metric taking `--layer-dir` can be pointed at. Attacks every metric here at once rather than one at a time. | minutes, CPU | **DONE**, delivered 2026-08-10 |
| 2 | **`RUN_EXTRA_METRICS.md`** — three further published metrics (Adams, Li, SAEBench) on the same inputs, so the failure can be shown not to be specific to ours. | ~1 h CPU | **DONE**, 96 cells, 0 failures |
| 3 | **`RUN_INTERPLM_ATTACK.md`** — InterPLM's *own published* metric on our models: their code, our backbone. Builds its own venv, does not touch `RUN_MUSTRUNS.sh`. | ~2–3 h setup + ~1–2 days grid | **DONE** — 81/81, delivered 2026-08-14 |
| 5 | **`RUN_CHECKS.md`** — the verification queue. **Run this next.** Grid audit, larger permutation null, fold-disjoint split, prevalence floors, global-level probe. | ~3 h, four of five stages CPU-only | **NEXT** |
| 4 | **`RUN_500TPP_SEEDS.md`** — takes the 500 tok/param budget table off n=1. | ~200 GPU-h | **STILL ON HOLD — do not start. See below.** |
| 6 | **`RUN_TIER1.sh`** — the paper's remaining pre-submission queue. Resumable, packages its own archive. | ~19 h, almost all CPU | **NEW (2026-08-31)** |

### Job 6 — the paper's remaining queue (added 2026-08-31)

Run these in this order, cheapest first, so a long one never blocks a short one. Every line is
resumable and skips finished work, so interrupt freely and re-run. Nothing here retrains a model
or refits a dictionary.

```bash
git pull

ONLY=1 bash RUN_TIER1.sh                      # minutes — builds the selectivity CSV, then reads it
ONLY=1 bash RUN_DEPTH_GRID.sh                 # 75 min  — corpus control at the other six depths
ONLY=2 bash RUN_TIER1.sh                      # 45 min  — fixed and rank-based denominator
ONLY=2 bash RUN_DEPTH_GRID.sh                 # <1 h    — probes at block 18
STAGE=11 bash RUN_MUSTRUNS.sh                 # ~1 h    — pairwise contact probes, SAE and raw
NSHUF_HI=100 ONLY=2 bash run_checks.sh        # 100-permutation null
FOLDDISJ_APPLY=1 ONLY=3 bash run_checks.sh    # fold-disjoint refit, native arm
ONLY=3 bash RUN_TIER1.sh                      # 12 h CPU — d_struct at three model seeds
ONLY=4 bash RUN_TIER1.sh                      # 6 h     — d_struct untrained baseline
bash RUN_BLOCKSHUFFLE.sh                      # ~6 h    — second destruction procedure
```

`RUN_BLOCKSHUFFLE.sh` is the only line that trains anything, and it is last because it is the
longest. It is also the one that closes the paper's first stated limitation. It was previously
costed as a multi-day job; it is not. The corpus takes an hour and a 42M model takes 47 minutes at
the 233k tok/s measured on this box, so six models and eighteen metric cells come to about six
hours. `SEEDS=42 bash RUN_BLOCKSHUFFLE.sh` does it at one seed in about three. It refuses to start
if its paths would touch the real corpus or the real checkpoints.

`bash RUN_TIER1.sh --plan` prints the plan and the cost without running anything.

**If you hit the selectivity CSV bug:** fixed. No committed script had ever built the *native*
`results_rigor/aa_selectivity.csv` — `RUN_MUSTRUNS.sh` stage 2 only builds the shuffled one, and
`experiment_aa_selectivity.py` defaults to `--root outputs_outlier` with the ESM-2 / RITA pair,
which is not the pair this paper reports. Stage 1 now builds it itself from `outputs_ctrl` with
the right arms, and skips cells that have no `Z.npy`. `SEL_SEEDS="42 43 44"` widens it beyond
seed 42 if you want more cells.

**Two things to send back that need no compute:**

1. `outputs_ctrl_folddisj/*/*/struct_seq_metrics.csv` — the *native* fold-disjoint cells. They
   already exist here; they were just not in the last archive. Without them the fold-disjoint
   shuffled numbers have nothing to be divided by.
2. `git add` the untracked scripts on this box and push — `label_features.py`,
   `experiment_interplm_metric_dsgate.py`, `patch_dsgate.py`, `patch_raw_coact*.py`,
   `run_queue_0804.sh`, `run_500tpp.sh`, `run_ppl500.sh`, `run_shuf500_eval.sh`, `unwrap_sae.py`,
   `merge_stage5.py`, `extract_esm2.py`, `interplm_sae_encode.py`. Five of them produced published
   numbers and exist in one copy.

**Still not authorised:** `RUN_500TPP_SEEDS.md` (job 4, ~200 GPU-h) and retraining the shared
dictionaries at blocks 11/14/18 (days, for a tidier table). Both are poor trades against what is
above.

---

### Job 3 is done. Run `RUN_CHECKS.md` next — still not job 4

```bash
git pull
bash run_checks.sh
```

Job 4 has not been unblocked and nothing about it has changed. `RUN_CHECKS.md` is what the
paper needs next and it costs about three hours against job 4's four to eight days.

The grid completed on 2026-08-14: 81/81 SAEs across five conditions, archive checksum
verified. `python check_grid.py` re-checks completeness and arm comparability in seconds and is
free to run at any time.

### Run 1–3, then STOP and report

**Do not start job 4.** Finish 1–3, send the results, and wait for a reply before touching
anything in `RUN_500TPP_SEEDS.md`. That brief is deliberately still in the repo so you can
read the plan, but it is not authorised to run yet.

This is not about the cost alone. Three reasons, so you can tell if the situation has
changed enough to ask again:

1. **1–3 create new results; 4 tightens an error bar on a result we already have.** There
   is a preprint on the critical path, so new evidence is worth more right now than a
   better bound on old evidence. The n=1 caveat on the budget table can be stated honestly
   in the meantime.
2. **4 monopolises the GPU for 4–8 days.** Jobs 1 and 2 are CPU-only and job 3 is 54 GPU
   embedding passes followed by ~13 h of single-threaded CPU — so 1–3 barely touch the
   GPU, while 4 owns it completely and nothing else moves.
3. **What job 4 *should be* depends on what job 3 says.** `RUN_500TPP_SEEDS.md` has a
   ranked fallback — shuffled-pair-only or masked-arm-only, ~50 GPU-h instead of ~200.
   Which cut is right depends on whether InterPLM's own metric turns out to be
   arm-dependent the way ours is. Deciding before job 3 lands risks spending 8 days on the
   wrong slice.

The likely instruction afterwards is the cheapest slice — **shuffled pair, seed 43 only,
~50 GPU-h** — rather than the full version. Don't pre-empt it; the point of stopping is
that the answer might be the masked arm instead.

Deadline for context: ICLR 2027 main track, **2026-09-24**. Jobs 1–3 are about two days,
so there is roughly a month of slack. Nothing here needs to be rushed.

1 and 2 both need `Z.npy` present in the layer dirs. If it was pruned,
`STAGE=1 bash RUN_MUSTRUNS.sh` rebuilds it (~2 h) and is the only expensive part of either.

Read 1–3 after the four commands below.

```
git pull
```

Then the four commands below, in this order. Each one is independent — if one goes
red, the others still work.

---

## 1. Fast, do this first (10 min, no GPU)

```
STAGE=6 bash RUN_MUSTRUNS.sh
```

Send back: `results_synthetic_composition/`

**Expected oddity:** the numbers will be identical across all six cells. That is
correct, not a bug — these features don't depend on the model. If the six cells
*disagree*, something is wrong and I need to know.

---

## 2. Also fast (30 min, no GPU)

```
STAGE=9 bash RUN_MUSTRUNS.sh
```

Send back: `results_nshuffle_sensitivity/`

**If it says `Z.npy missing`:** run `STAGE=1 bash RUN_MUSTRUNS.sh` first (that
rebuilds Z, about 2 h), then re-run stage 9.

---

## 3. GPU, about 2 h

```
STAGE=7 bash RUN_MUSTRUNS.sh
```

Send back: `outputs_ctrl_dseed43/` and `outputs_ctrl_dseed44/`
(the `struct_seq_metrics.csv` and `META.json` from each cell — you can drop `Z.npy`,
it's huge and I don't need it)

Needs the checkpoints at `~/own_sae_data/uniref50_pilot/`. If they've moved, tell me
rather than hunting for them.

---

## 4. GPU, about 1 h

```
STAGE=8 bash RUN_MUSTRUNS.sh
```

Send back: `results_rigor/capacity_vs_lstruct_extended.csv`

---

## 5. The reviewer-driven batch (added 2 Aug)

Four more stages, all from supervisor feedback. Run them in this order.

```
STAGE=10 bash RUN_MUSTRUNS.sh     # ~1 h GPU  -- raw L_struct + perplexity
STAGE=14 bash RUN_MUSTRUNS.sh     # ~2 h CPU  -- Simon & Zou's metric, exactly
STAGE=12 bash RUN_MUSTRUNS.sh     # ~1 h CPU  -- matched contact-map nulls
STAGE=11 bash RUN_MUSTRUNS.sh     # ~1 h CPU  -- pairwise probes (needs 10 first)
STAGE=15 bash RUN_MUSTRUNS.sh     # ~1 h GPU  -- matched untrained baseline
STAGE=13 bash RUN_MUSTRUNS.sh     # ~2 h CPU  -- continuous separation sweep
```

Order matters in one place: **10 before 11**. Stage 10 writes the raw activations
that stage 11's raw arm needs, and that arm is the half that answers the reviewer.
If you run 11 first it will say so in red rather than skipping quietly.

Then, whatever you ran:

```
.venv/bin/python summarize_reviewer_batch.py
```

Send me its output too — it does all the comparisons, so neither of us has to do
them by hand. That is where the mistakes have been.

Send back: `results_raw_coactivation/`, `results_contact_null/`,
`results_pairwise_probe/`, `results_sep_continuous/`

Stage 10 is the important one. Stages 11-13 need `Z.npy` in `outputs_ctrl`; if they
report it missing, run `STAGE=1` first.

**One thing to check rather than trust:** stage 12 prints a `swap rate`. If it comes
back near zero the null didn't actually happen and the number is meaningless. It
should be around 0.97. I broke this once already and it failed silently.

## How to tell if it worked

Green `OK` lines and the script exits 0. It exits non-zero on any failure and prints
a summary of what broke, so "it went green" is a reliable signal — you don't need to
read the output.

If anything goes red: send `logs_mustruns/` and don't try to fix it.

## One thing that changed in the code

`eval_ctrl_saefree.py` now pins the random seed on the linear probe. If you re-run
stage 4 the probe numbers may shift in the last digit or two. That's expected and is the point — say so if you notice it, rather than treating it as a regression.

## One check that is not optional

Stage 12 prints a `swap rate` and stage 11 prints a `POSITIVE CONTROL` line. Both
exist because the thing they guard has already failed silently once.

- stage 12 swap rate should be around **0.78**. Near zero means the null never
  happened and the numbers are meaningless.
- stage 11 positive control should land near **0.61**. Near 0.50 means the probe
  itself is too weak, and every other number from that stage is uninterpretable.

