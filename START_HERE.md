# Start here

Ignore the other RUN_*.md files. They are older and some of them contradict this one.

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
STAGE=10 bash RUN_MUSTRUNS.sh     # ~1 h GPU  -- raw-dimension L_struct + perplexity
STAGE=12 bash RUN_MUSTRUNS.sh     # ~1 h CPU  -- matched contact-map nulls
STAGE=11 bash RUN_MUSTRUNS.sh     # ~1 h CPU  -- pairwise contact probes
STAGE=13 bash RUN_MUSTRUNS.sh     # ~2 h CPU  -- continuous separation sweep
```

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

