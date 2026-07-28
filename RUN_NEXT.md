# 👉 Everything left, one command — 2026-07-28

You last ran `b8919e8`. Since then I've pushed one thing: a script that chains all the
remaining work so you don't have to run them one at a time.

```bash
git pull
bash run_all_pending.sh
```

**~14–19 h. Start it and walk away.** That's the whole instruction.

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
| 3 | The other fair-compute protocol at seeds 43/44 — closes our last single-seed claim | 1.5 h |
| 4 | Does the result depend on how we define a contact? (distance × separation) | 1 h |
| 5 | Does it hold with no sparse autoencoder involved at all? | 2 h |
| 6 | Does our metric depend on autoencoder quality at all? | 2.5 h |
| 7 | Crosscoder — 15-min viability check first, full run only if it passes | 1–2 h |

Stage 1 is the one I most want an answer to. Stage 7 gates itself: if the quick check says the
measurement won't have enough range to support a claim, it stops there and says so — that's
intended, not a failure.

## If you want finer control

```bash
ONLY=3 bash run_all_pending.sh      # run just one stage
SKIP="7" bash run_all_pending.sh    # everything except the crosscoder
```

Each stage is standalone — stage 4 rebuilds what it needs if you run it on its own.

## Two asks

- **Keep `~/own_sae_data/`.** Every checkpoint gets reused.
- **Send `logs_pending/` back with the results.** The script prints the exact `tar` command
  when it finishes. The logs carry the git revision, which is how we know which code produced
  which number — that's been genuinely useful twice now.

## If it dies immediately

The only hard stop is a missing `cache/pdb_files`. Everything past that is soft-failing. Send
the last 50 lines plus `logs_pending/` and I'll sort it.
