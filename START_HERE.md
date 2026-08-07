# Start here

Ignore the other RUN_*.md files. They are older and some of them contradict this one.

**Two exceptions, both new (2026-08-07) and not superseded by anything here:**

- **`RUN_INTERPLM_ATTACK.md`** — runs InterPLM's *own published* metric on our models:
  their code, our backbone. Builds its own venv, does not touch `RUN_MUSTRUNS.sh`.
- **`RUN_NOMODEL_BASELINE.md`** — builds one layer dir whose features contain no model,
  which any metric taking `--layer-dir` can then be pointed at. Cheap, and it attacks
  every metric here at once rather than one at a time.

Read both after the four commands below.

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

