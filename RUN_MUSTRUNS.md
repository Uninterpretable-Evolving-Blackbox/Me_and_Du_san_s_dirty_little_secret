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

**Run the metric with InterPLM's contact definition: 6 Å, no separation floor.**

### What their method actually is (read off the paper, §5.4.2, verbatim)

> 1. Identified high-activation regions (>0.6) in proteins with available AlphaFold structures
> 2. For each protein's **highest-activation residue**, calculated: Sequential clustering: mean
>    activation within ±2 positions in sequence; Structural clustering: mean activation of residues
>    **within 6Å in 3D space**
> 3. Generated null distributions through averaging **5 random permutations** per protein
> 4. Assessed clustering significance using paired *t*-tests and **Cohen's d** across 100 proteins
>    per feature

### Where we match them, and where we do not

| | InterPLM | ours |
|---|---|---|
| structural cutoff | **6 Å** | 8 Å (Cα–Cα) |
| **separation floor** | **none** | **≥ 12** |
| sequential baseline | ±2 | ±1, ±2 |
| permutations | 5 | 5 |
| effect size | Cohen's *d* | Cohen's *d* |
| **anchor** | the **single highest-activation residue** per protein | **all** residues where the feature is active |
| **activation threshold** | absolute **> 0.6** | top-10% fraction (`topk_frac`) |
| **unit of test** | per feature, across 100 proteins | pooled across features |
| structures | AlphaFold | SCOPe / experimental PDB |
| inclusion | ≥ 25 examples per feature, Bonferroni *p* < 0.05 | ≥ 5 active residues per (feature, protein) |

**So this stage is "our estimator at their contact definition", not "their metric".** The two
differ in the anchor, the threshold and the unit of test as well as the cutoff. That is worth being
exact about, because it bounds what the result can be used to say.

### What it does answer, which is the question the paper needs

Our `L_struct` adds a minimum sequence separation that InterPLM does not use, and the sweep showed
that floor carries **81–93%** of the whole effect. Stage 1 isolates that one variable:

| result at 6 Å / gap 1 | reading |
|---|---|
| shuffled/real ≈ 46× | the failure survives without our floor — it is a property of the co-activation construction |
| shuffled/real ≈ 2× | the failure **tracks our separation floor** — "making the metric stricter is what made it invalid" |

Prediction, recorded before the run so it cannot be retrofitted: **the second**. At 6 Å with no
floor, *i*±1 (≈3.8 Å) and most *i*±2 qualify as structural neighbours, so the structural measure
largely overlaps its own sequential baseline and should inherit that baseline's milder behaviour.

**What it cannot tell us** is whether InterPLM's own published analysis is affected. That would need
their estimator too — single anchor, absolute 0.6 threshold, per-feature test over AlphaFold
structures — which is a separate build and a much larger claim. Not attempted here.

The `--gaps 1,2,12` grid gives their setting (1), a near-neighbour check (2), and our default (12)
on one axis, so the dose–response is visible in a single CSV.

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

## Estimated wall time, from your own logs

Grounded in `pending_core_20260729/logs_pending/stage4_*.log` (162 parallel blocks, 22.5 min of
compute on 32 cores for 81 sweep cells) and the measured ~9 min per SAE training.

| stage | work | estimate |
|---|---|---|
| **0** | rebuild `Z.npy` for 12 cells | **~2 h** |
| **1** | 18 sweep cells, 1 cutoff x 3 gaps x 3 layers x 2 roots | **25–45 min** |
| **2** | 2 cells, no SAE training | **< 5 min** |
| **3** | 54 cells, trains an SAE per cell | **7–9 h** |
| **4** | 6 arms x 9 depths, probes only | **1–2 h** |
| **5** | corpus prep + 4 PLM training runs | **4–5 h** |

**Stage 0 is the correction.** `prune_z` deletes `Z.npy` after every pipeline run, so the controlled
and shuffled trees almost certainly have none, and stage 1 cannot run without it. Rebuilding costs an
SAE training per cell, so **the real cost of the stage-1 answer is ~2.5–3 h, not 45 min.** An earlier
version of this file said 25–45 min; that was the sweep alone and it was wrong.

Regeneration is safe and has been done before (`fix_pred_bootstrap.sh`): the SAE seed and the val
split are both recorded, so it reproduces the same dictionary — last time to the last digit — and it
does not write `struct_seq_metrics.csv`, so delivered results are untouched.

**If time is short: stages 0, 1 and 2.** About three hours, and they answer the question the paper
turns on. Stage 3 is the one to decide on deliberately — 7–9 h to move a correlation from n=6 to
n=54.

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
