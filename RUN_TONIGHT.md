# 👉 Run these two before bed — 2026-07-27

Thanks for the seeds run; it came back clean and the effect replicated across all three.
Everything below is new code pushed today. **Nothing here needs you awake** — both jobs are
resumable and skip finished work, so if the machine sleeps or you interrupt them, just re-run.

```bash
git pull

# 1. ~2-4 h — analysis only, no training.
STAGE=2 bash run_ctrl_rigor.sh

# 2. ~3 h — trains 2 small models. Start it when 1 finishes.
STAGE=2 bash run_ctrl_mechanism.sh
```

If you only have time to start one, **start 1** — job 2 depends on the answer.

You can run them back to back in one go if you prefer:

```bash
git pull && STAGE=2 bash run_ctrl_rigor.sh && STAGE=2 bash run_ctrl_mechanism.sh
```

---

## What each one is for

**Job 1 — the SAE capacity sweep.** We found a real problem: the sparse autoencoder used on
the controlled models keeps 256 of 320 dimensions active, i.e. 80%. On ESM-2 the same setting
is 20%. At 80% it barely compresses anything, which is almost certainly why the masked model's
reconstruction sits at 0.99 in the shallow layers and why it fits better than the causal model
at every depth. This sweeps k and the expansion factor and tells us which setting puts both
arms in a comparable regime. `k=64` is the predicted answer.

**Job 2 — the shuffled-sequence control.** Our headline metric is homemade, and nothing so far
shows it goes to zero when it should. This trains the same two models on a corpus where each
sequence's residues have been shuffled — same length, same amino-acid composition, all order
destroyed — then evaluates them on the real proteins and real structures. If the metric is
measuring structure, it should collapse. If it doesn't collapse, we need to know that before
we write anything.

---

## Please don't run these yet

`STAGE=1 bash run_ctrl_mechanism.sh` (the directional test) and `STAGE=1 bash run_ctrl_rigor.sh`
(frozen baselines). Both depend on what the sweep in job 1 decides, so they're next, not now.
`run_crosscoder_ctrl.sh` is still on hold.

## Two things to keep

- **Keep `~/own_sae_data/`.** Every checkpoint gets reused.
- **Don't let `Z.npy` get deleted for the moment.** Some analyses we're about to run need it,
  and it can only be rebuilt by retraining the autoencoders. If you run anything by hand, pass
  `KEEP_Z=1`.

## What to send back

Job 1 prints a table and writes a CSV; job 2 writes to `outputs_ctrl_shuf/`. Same as last time:

```bash
tar czf tonight_results.tgz \
  $(find outputs_ctrl_shuf -name 'struct_seq_metrics.csv' -o -name 'META.json') \
  results_ctrl_sweep* outputs_robustness/*.csv *.log
```

**Please include the log this time** — last run's tarball didn't have `train.log`, and the log
header is what tells us which revision and which settings actually ran. It also means we can
tell whether a number came from the code we think it did.

## If something breaks

Nothing here overwrites existing results: job 1 is read-only analysis, job 2 writes to its own
corpus directory (`~/own_sae_data/uniref50_pilot_shuf`) and its own output root
(`outputs_ctrl_shuf`). So a failure costs time, not data. Send the last ~50 lines of output and
we'll sort it out.
