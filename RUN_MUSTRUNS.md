# RUN_MUSTRUNS — what's left, and why each one matters

```bash
git pull
bash RUN_MUSTRUNS.sh          # stages 1-4, no training
STAGE=5 bash RUN_MUSTRUNS.sh  # the long one, only if you have the hours
```

Idempotent (skips anything already produced), deletes nothing, logs to `logs_mustruns/`, and
**exits non-zero if any stage fails** so a wrapping queue can't print OK over a dead stage.

---

## Stage 1 — the one that matters most

**Run the metric at Simon & Zou's own settings: 6 Å, no separation floor.**

Our `L_struct` adds a minimum sequence separation (`|i-j| >= 12`) that InterPLM does not use. The
contact-definition sweep showed that floor carries **81–93%** of the entire effect. So we do not
currently know whether the shuffled-input failure belongs to the *published recipe* or specifically
to *our modification of it*.

The two possible answers are different papers:

| result at 6 Å / gap 1 | what it means |
|---|---|
| shuffled/real ≈ 46× | the published recipe fails too — the claim broadens |
| shuffled/real ≈ 2× | the failure tracks **our** separation floor — the claim narrows, and the finding becomes "making the metric stricter is what made it invalid" |

My prediction is the second: at 6 Å with no floor, *i*±1 (≈3.8 Å) and most *i*±2 qualify as
structural neighbours, so their structural measure largely overlaps its own sequential baseline and
should inherit that baseline's much milder behaviour. **Recording the prediction here so it can't be
retrofitted after the fact.**

This needed a code change — the sweep hard-coded its grid. `--cutoffs` and `--gaps` are new;
defaults are unchanged, so every number already delivered reproduces bit-for-bit.

## Stage 2 — amino-acid selectivity on the *shuffled* checkpoints

Shuffling permutes rather than resamples, so amino-acid composition is the one thing it preserves
**exactly**. "The inflation is composition-driven" is therefore the live alternative explanation —
and it has only ever been tested on the published ESM-2/RITA pair, never on the models it actually
concerns. `experiment_aa_selectivity.py` is new to this repo (it existed only on the laptop).

## Stage 3 — regenerate `val_EV` at every depth and seed

`val_EV` was logged for the real models at **three depths only**, so the reconstruction-gain
correlation rests on n=6 (3 depths × 2 arms, seed 42). This is **not** a re-analysis: the metadata
does not exist for the other cells, so it has to be recomputed from activations. Uses the existing
`measure_rank_ev.py`. Takes n=6 → n=54.

## Stage 4 — the MLP probe

Your `--mlp` code from `c169023`, never run. This is a **falsification** test, not a robustness
check: if the depth reversal is a fact about linear separability rather than information content,
the probe half of the paper loses its claim. Worth knowing before submission rather than after.

## Stage 5 — two more shuffled training runs (seeds 43, 44)

The shuffled control and the `L_seq` positive control are both seed-42-only, because only one
shuffled pair was ever trained. Long; not run by default.

---

## Traps the script handles for you

- **`--n-shuffles 5`, never the default 3.** Three computes a different metric than every other
  number in the project.
- **`KEEP_Z=1`** wherever `Z.npy` is needed downstream — `prune_z` has silently removed a later
  stage's input before.
- **Exit codes are checked and aggregated.** The previous queue printed `STAGE 3 OK` while both of
  its bootstraps were dead; this one cannot.

## Not in here, deliberately

Three controls (directional, dictionary-capacity, contact-definition) should also be run on the
**published ESM-2/RITA pair** — none ever has been, and it's the largest coverage gap. Those
activations are on the laptop, not this box, so that job belongs there.

## Changed in this commit

| file | change |
|---|---|
| `experiment_contact_def_sweep.py` | **patched** — added `--cutoffs` / `--gaps`. Defaults untouched |
| `experiment_aa_selectivity.py` | **new** — copied from the laptop, self-contained |
| `RUN_MUSTRUNS.sh` | **new** — the five stages |
| `RUN_MUSTRUNS.md` | **new** — this file |
