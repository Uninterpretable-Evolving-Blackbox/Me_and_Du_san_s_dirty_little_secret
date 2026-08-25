# Rescoring batch

Three stages over checkpoints that already exist. No pretraining, no new models,
single GPU, roughly two hours in total.

```bash
git pull
./preflight.sh all          # a couple of minutes, no GPU
bash run_gaps.sh            # all three stages
```

If `preflight.sh` fails, stop and report it rather than working around it.

---

## Before you start

Two checkpoint roots must exist. The script warns and skips cells if they don't:

| variable | default | holds |
|---|---|---|
| `NATIVE_ROOT` | `~/own_sae_data/uniref50_pilot` | native-corpus checkpoints |
| `SHUF_ROOT` | `~/own_sae_data/uniref50_pilot_shuf` | shuffled-corpus checkpoints |

Override on the command line if they live elsewhere:

```bash
SHUF_ROOT=/data/pilot_shuf bash run_gaps.sh
```

**Stage 3 additionally needs `eval_set_folddisj/`.** If it isn't there:

```bash
FOLDDISJ_APPLY=1 ONLY=3 bash run_checks.sh
```

Every stage skips cells whose `struct_seq_metrics.csv` already exists, so
interrupting and re-running costs nothing. A stage that produces **no** cells
exits non-zero rather than reporting success — if you see that, a checkpoint
root is wrong.

Run stages individually with `STAGE=1`, `STAGE=2`, `STAGE=3`.

---

## Stage 1 — permuted-input rescore (18 cells, ~40 min)

```bash
STAGE=1 bash run_gaps.sh
```

Builds `eval_set_evalshuf/` — the same 1,500 domains with residue order permuted
inside each sequence, same uids, same lengths, same validation split — and
rescores the **shuffled-corpus** checkpoints on it at blocks 11, 14 and 18,
three seeds per arm.

Structures, contact graphs and scoring code are unchanged from the existing run.
The only difference is the evaluation input.

Output → `outputs_ctrl_evaldist/<arm>/layer_<L>/struct_seq_metrics.csv`

---

## Stage 2 — random-init at seeds 43 and 44 (12 cells, ~30 min)

```bash
STAGE=2 bash run_gaps.sh
```

Runs the existing random-initialisation condition at two further seeds, both
arms, blocks 11/14/18. `--randomize-model` discards the trained weights, so the
checkpoint is read for its architecture only — initialise, one forward pass, fit
the dictionary. No pretraining.

Output → `outputs_ctrl_randominit_s43/…` and `outputs_ctrl_randominit_s44/…`

---

## Stage 3 — fold-disjoint rescore (18 cells, ~40 min)

```bash
STAGE=3 bash run_gaps.sh
```

`run_checks.sh` stage 3 has its checkpoint root hardcoded to `uniref50_pilot`,
so it covers the native arm only. This runs the shuffled-corpus arm against the
same `eval_set_folddisj`, at the same blocks and seeds.

Output → `outputs_ctrl_folddisj_shuf/<arm>/layer_<L>/struct_seq_metrics.csv`

---

## What to send back

```
outputs_ctrl_evaldist/*/*/struct_seq_metrics.csv
outputs_ctrl_randominit_s43/*/*/struct_seq_metrics.csv
outputs_ctrl_randominit_s44/*/*/struct_seq_metrics.csv
outputs_ctrl_folddisj_shuf/*/*/struct_seq_metrics.csv
eval_set_evalshuf/META.json
```

Plus, from this directory:

```bash
git rev-parse HEAD
git status --short
```

The commit ties the numbers to the code that produced them; `git status` catches
uncommitted local edits, which the hash alone cannot.

**Do not send** `.npy`, `.pt` or checkpoint files. They are large and nothing
downstream needs them.

---

## If something goes wrong

- **`[skip] no <path>`** — a checkpoint root is wrong. Set `NATIVE_ROOT` or
  `SHUF_ROOT` and re-run; completed cells are not redone.
- **`NO cells produced`** — the whole stage found nothing. Same cause, and the
  stage exits non-zero so it can't be mistaken for a clean run.
- **`no eval_set_folddisj/`** — run
  `FOLDDISJ_APPLY=1 ONLY=3 bash run_checks.sh` first.
- **A cell fails partway** — the stage continues and returns non-zero at the
  end. Re-running picks up where it stopped.
- **Anything else** — send the console output rather than diagnosing it.

---

## Also available, not in this batch

Three further items need a code change or a training run rather than a rescore,
so they are excluded here:

1. Recomputing the statistic with a fixed and with a rank-based denominator —
   a change inside `cpu_stage.py`.
2. A second corpus-destruction procedure (block shuffle preserving local
   windows) — two pretraining runs at 42M parameters.
3. The random-init condition for the reimplemented comparison statistic.

Say if you would rather have one of these than one of the three above.
