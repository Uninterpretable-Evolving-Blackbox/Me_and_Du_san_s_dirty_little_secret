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

## How to tell if it worked

Green `OK` lines and the script exits 0. It exits non-zero on any failure and prints
a summary of what broke, so "it went green" is a reliable signal — you don't need to
read the output.

If anything goes red: send `logs_mustruns/` and don't try to fix it.

## One thing that changed in the code

`eval_ctrl_saefree.py` now pins the random seed on the linear probe. If you re-run
stage 4 the probe numbers may shift in the last digit or two. That's expected and is
the point — say so if you notice it, don't treat it as a regression.
