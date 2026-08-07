# The no-model baseline — one input, every metric

```bash
git pull

# build a layer dir whose "features" contain no model at all
python make_synthetic_layer.py --src outputs_ctrl/ckpt_mlm_s42_token/layer_14 \
                               --out outputs_synthetic/composition
```

Then point **any** metric that takes `--layer-dir` at it. Nothing downstream needs to
change, and no metric code is edited.

Send back: whatever each metric writes, plus the `occupancy` block from
`outputs_synthetic/composition/META.json`.

---

## What it does

Every metric here reads a layer directory: `Z.npy` plus `uids.json`, `lengths.npy`,
`offsets.npy`, `sequences.json`. `make_synthetic_layer.py` writes one where the 29
"features" are **indicator functions of residue identity** — 20 single amino acids,
8 chemical classes, and one all-ones column:

```
aa:A … aa:Y            1 where the residue is that type, else 0
class:hydrophobic …    1 where the residue is in AVLIMFWC, etc.
TRIVIAL:all-ones       1 everywhere  (the prevalence floor for any precision/recall metric)
```

No model. No SAE. No training. A metric cannot tell whether that column came from a
trained dictionary or from an if-statement — so if the dumb column scores like a learned
feature, the metric is not measuring learned structure.

## Why this is the cheap way to attack every metric at once

We already do this for `L_struct` (`experiment_synthetic_composition.py`: `aa:C` scores
**+0.4221** and beats all 2,560 learned features in 5 of 6 cells). Generalising the
*input* rather than editing each *metric* means one artefact covers all of them:

| metric | script | what a high synthetic score would mean |
|---|---|---|
| `L_struct` | `cpu_stage.py` | already known to fail: +0.4221 |
| InterPLM structural clustering | `experiment_interplm_metric.py` | **already run: `aa:C` reaches only +0.146, p=0.149 — their estimator resists it.** Their single-anchor-per-protein rule never aggregates a composition signal, where averaging over every active residue does |
| concept-F1 | `experiment_concept_f1.py` | a residue-type floor under every concept |
| AA selectivity | `experiment_aa_selectivity.py` | sanity check — should score at ceiling by construction |

That contrast is worth keeping: a critique in which every published metric conveniently
fails reads as a broken pipeline. One of them resisting is what makes the rest credible.

## Two traps, both measured

**1. Use `--jitter 1e-3` for anything that picks a single anchor.**
Binary features tie everywhere, so `np.argmax` returns the *first* index and the anchor is
pinned to the N-terminus. Measured mean relative anchor position: **0.009** for
`class:hydrophobic` against **~0.5** for a real SAE feature. That artefact alone produced
a spurious "the estimator rejects composition" result — mean `d_struct` −0.183 with 5
significant features — which vanished to −0.042 and **0 significant** once the ties were
broken. The jitter is added only where the indicator is 1, so the active set and any
threshold gate are unchanged.

**2. Use `--no-trivial` for anything with a top-k gate.**
`cpu_stage._cohens_d_vectorized` selects actives via `acts > percentile(acts, 100*(1-topk_frac))`.
A binary feature whose occupancy exceeds `topk_frac` has percentile exactly 1.0, nothing
is strictly greater, and the result is silently **0**. The all-ones column has occupancy
1.0 and always hits this. It is meaningful only for precision/recall metrics such as
concept-F1, where it is the prevalence floor.

The script prints an occupancy table and warns when any feature exceeds a given
`topk_frac`, so this is visible rather than silent.

## The general rule this encodes

Give the metric an input whose meaning you already know, and see what it says. It costs
one artefact per evaluation set, works on metrics you did not write, and needs no
agreement from their authors about what their code does.
