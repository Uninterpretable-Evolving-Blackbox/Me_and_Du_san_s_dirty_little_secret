# 👉 Everything left, one command — 2026-07-28

You last ran `b8919e8`. Since then I've pushed one thing: a script that chains all the
remaining work so you don't have to run them one at a time.

```bash
git pull
bash run_all_pending.sh
```

**~24–27 h. Start it and walk away.** That's the whole instruction.

> **Corrected 2026-07-28.** This file previously said ~14–19 h, with stage 3 at 1.5 h.
> That was wrong, and Ronnie caught it before starting. `--match-predictions` divides the
> step count by the mask rate (`train_ctrl_plm.py:165`), so the prediction-matched MLM arm
> trains on 660M / 0.15 = **4.4B tokens**, ~5.25 h per model. Both seeds need training from
> scratch — the CLM checkpoints are shared across protocols and skip — so stage 3 is ~12 h,
> not 1.5 h. The earlier figure came from applying the token-matched rate. Everything else
> in the queue is unchanged.

---

## It's built so you never have to babysit it

- **Skip-if-done everywhere.** Interrupt it, reboot, lose power — re-run the exact same
  command and it carries on, re-doing at most the one cell that was in flight.
- **A failing stage doesn't stop the queue.** Stages don't depend on each other, so a crash
  at 3am can't leave the GPU idle till morning. It logs the failure, moves on, and prints a
  summary of what failed at the end.
- **Nothing overwrites anything.** Each stage writes to its own output directory.
- **No settings to change.** The defaults are what we want.

## The seven stages

| # | what it answers | ~time |
|---|---|---|
| 1 | Does the masked model's advantage sit on contacts a causal model literally cannot see? | 2 h |
| 2 | Does a random, untrained dictionary reproduce the controlled result too? | 3–5 h |
| 3 | The other fair-compute protocol at seeds 43/44 — closes our last single-seed claim | **12 h** |
| 4 | Does the result depend on how we define a contact? (distance × separation) | 1 h |
| 5 | Does it hold with no sparse autoencoder involved at all? | 2 h |
| 6 | Does our metric depend on autoencoder quality at all? | 2.5 h |
| 7 | Crosscoder — 15-min viability check first, full run only if it passes | 1–2 h |

Stage 1 is the one I most want an answer to. Stage 7 gates itself: if the quick check says the
measurement won't have enough range to support a claim, it stops there and says so — that's
intended, not a failure.

**Stage 3 is half the queue and four times the cost of anything else**, while being third in
value. If you are starting fresh rather than mid-run, `SKIP="3"` gets the other six done in
~12–15 h and stage 3 can follow on its own with `ONLY=3`. Same total compute, but the
high-value answers arrive a day earlier. If it is already running, leave it — nothing
downstream depends on the order.

## If you want finer control

```bash
ONLY=3 bash run_all_pending.sh      # run just one stage
SKIP="7" bash run_all_pending.sh    # everything except the crosscoder
```

Each stage is standalone — stage 4 rebuilds what it needs if you run it on its own.

## Two things worth knowing

- **Disk.** Z files are ~1.5 GB per cell. Stage 1 keeps ~27 GB and stage 3 runs with
  `KEEP_Z=1` for another ~27 GB, so expect ~54 GB to accumulate and stay until we prune it
  deliberately. Far short of the 500 GB that filled the disk in July, but worth a glance if
  you are tight.
- **Stage 1 rewrites `struct_seq_metrics.csv`** on the 18 cells it touches (3 seeds x 2 arms
  x depths 11/14/18). It regenerates Z from the same checkpoint with the same fixed SAE
  seed, so it should reproduce exactly — but those are the C5 headline depths, so we diff
  them against the archived originals when the results land. Nothing is at risk; we kept a
  copy of all 18.

## Two asks

- **Keep `~/own_sae_data/`.** Every checkpoint gets reused.
- **Send `logs_pending/` back with the results.** The script prints the exact `tar` command
  when it finishes. The logs carry the git revision, which is how we know which code produced
  which number — that's been genuinely useful twice now.

## If it dies immediately

The only hard stop is a missing `cache/pdb_files`. Everything past that is soft-failing. Send
the last 50 lines plus `logs_pending/` and I'll sort it.
