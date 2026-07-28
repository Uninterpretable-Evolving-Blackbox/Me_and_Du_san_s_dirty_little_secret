# 👉 One command, everything outstanding — 2026-07-28

Last night's two jobs both came back clean, thanks. The sweep answered its question (no
setting equalises the two arms — that's the result, not a problem), and the shuffled-corpus
control told us something important about the metric.

Everything left is now chained into **one script**. Start it and forget it:

```bash
git pull
bash run_all_pending.sh
```

~14–19 h. Seven stages, run back to back.

## It is built to be left alone

- **Every stage is skip-if-done.** Interrupt it, reboot, re-run the same command — it
  continues from where it stopped and re-does at most the cell that was in flight.
- **A failing stage never stops the queue.** Later stages don't depend on earlier ones, so a
  crash at 3am can't idle the GPU until morning. Failures are collected and printed at the end
  with the last 25 lines of each log.
- **Nothing overwrites existing results.** Each stage writes to its own output root.

## What the seven stages are

| # | what | ~time |
|---|---|---|
| 1 | Upstream vs downstream contacts — the mechanism test | 2 h |
| 2 | Frozen-dictionary baselines on the controlled pair | 3–5 h |
| 3 | Prediction-matched protocol, seeds 43/44 | 1.5 h |
| 4 | Contact-definition sweep (Cα 6/8/10 × separation 6/12/24) | 1 h |
| 5 | SAE-free readout: linear probe + contact prediction, no SAE | 2 h |
| 6 | Does L_struct depend on SAE capacity at all? | 2.5 h |
| 7 | Crosscoder — 15-min gate first, full run only if it passes | 1–2 h |

Stage 1 is the one I care about most: it asks whether the masked model's advantage sits on
contacts a causal model structurally can't see. Stage 7 gates itself — if the 15-minute check
says the measurement won't have enough dynamic range, it stops there and says so. That's
designed behaviour, not a failure.

## No settings to change

The defaults are already what we want. If you need them:

```bash
ONLY=3 bash run_all_pending.sh      # just one stage
SKIP="7" bash run_all_pending.sh    # everything except the crosscoder
```

## Two asks

- **Keep `~/own_sae_data/`** — every checkpoint gets reused.
- **Send the logs back.** They're in `logs_pending/`, and the script prints the exact `tar`
  command at the end. The log header records the git revision, which is how we know which
  code produced which number.

## If it dies early

The only hard stop is a missing `cache/pdb_files`. Everything after that is soft. Send the
last 50 lines and the contents of `logs_pending/` and we'll sort it.
