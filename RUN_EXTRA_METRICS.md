# Three more published metrics, same attack

```bash
git pull

# one cell, to check it runs on your layout (30-60 s)
python experiment_extra_metrics.py \
    --layer-dir outputs_ctrl/ckpt_mlm_s42_token/layer_14 \
    --metric all --sae <that cell's ae.pt> \
    --out results_extra/mlm_s42_L14

# the no-model attack — note the --jitter, it is not optional here
python experiment_extra_metrics.py \
    --layer-dir outputs_synthetic/composition \
    --metric all --jitter 1e-3 \
    --out results_extra/synthetic
```

Send back: `results_extra/`. **CPU only, no GPU, no training.**

---

## Why

Everything we have shown attacks two measures: our `L_struct` and InterPLM's. The first
question a reviewer asks is whether the failure is specific to those two. This runs three
more, each from a different paper, on **identical inputs** — same layer dir, same concept
set, same fold-disjoint split. The only thing that varies is the metric, which is what
makes a difference between them attributable to the metric rather than the data.

| metric | paper | what it asks |
|---|---|---|
| `adams` | Adams et al., InterProt (ICML 2025) | single best latent per concept, plain per-residue F1, **no domain adjustment** |
| `geometry` | Li et al., *Entropy* 2025, 27, 344 | do co-activating features sit close in decoder space? vs their random-direction null |
| `ksparse` | SAEBench (Karvonen et al., ICML 2025) | can the top-k features linearly predict a concept? swept over k |

**`adams` is the one with a mechanism attached.** InterPLM's `calculate_f1.py` divides
recall by domain; Adams' does not. Running both on the same concepts isolates that
adjustment. We already know InterPLM's estimator *resists* the no-model baseline
(`aa:C` reaches only +0.146, p = 0.149). If Adams' does not resist it, the domain
adjustment is what protects them — which is a finding about why a metric survives, not
just another metric failing.

## What I already measured here

Run locally on layer 32 of the ESM-2 650M dictionaries in `outputs_layerwise/` and
`outputs_random/`, against the same 80 SCOPe/SS/RSA concepts:

| input | `adams` mean test F1 | concepts at or below the prevalence floor |
|---|---|---|
| synthetic, 29 features, **no model** | **0.0683** | 66/80 |
| trained ESM-2 SAE, 10,240 features | 0.0587 | 71/80 |
| randomised-weight ESM-2 SAE, 10,240 | 0.0580 | 72/80 |

Two things, and the second matters more than the first.

1. The no-model baseline scores **above** both, and trained sits ~0.0007 above randomised
   — Heap et al.'s trained-vs-random result, reproduced on a protein SAE with a concept
   metric rather than an auto-interpretability score.

2. **Do not quote this yet.** 71 of 80 concepts are at or below the prevalence floor for
   the trained dictionary, and every mean is 0.058–0.068. That is a regime with almost no
   discriminative power, so "synthetic beats trained" is a difference inside a range where
   nothing separates. This is the §4.4 lesson applied to our own result: an untrained or
   no-model baseline needs a **power check** before a null from it means anything. The
   power check here is a concept set on which the metric demonstrably works — which is
   what running it on our controlled pair will establish, or not.

`geometry` on the trained ESM-2 dictionary: ρ = 0.344 between co-activation Jaccard and
decoder cosine, against a random-direction null of −0.0003 ± 0.0016 (z ≈ 209). So Li's
effect is real and very far from their null on a real dictionary. The question this run
answers is whether the no-model input produces the same structure — amino acids co-occur
for chemical reasons, so a dictionary of residue detectors should show "lobes" too.

## Cost

Measured, not estimated — 295k residues:

| features | `adams` | `geometry` | `ksparse` | peak RSS |
|---|---|---|---|---|
| 10,240 | 62 s | 54 s | 28 s | 12.9 GB |
| 2,560 (our cells) | ~20 s | ~5 s | ~15 s | ~4 GB |

`geometry` scales as features², the other two roughly linearly. For the full sweep —
18 real cells + 18 shuffled + 12 at 500 tok/param + 3 untrained + synthetic — expect
**about an hour, single process.** It is the cheapest thing in the queue by a wide margin.

## Dependencies and traps

1. **Needs `Z.npy` in each layer dir.** If it was pruned, `STAGE=1 bash RUN_MUSTRUNS.sh`
   rebuilds it (~2 h). That is the only expensive part.
2. **`--jitter 1e-3` on the synthetic input, always.** Binary features tie everywhere and
   `argmax` then pins every anchor to the N-terminus. That artefact produced a spurious
   result once already.
3. **Memory is adaptive but not free.** Chunk size is derived from the feature count to
   hold arrays near 200 MB; it was 12.5 GB peak before that was fixed. Do not run more
   than 2–3 of these concurrently, and size against RAM, not cores.
4. **`--sae` is only needed for `geometry`.** Without it the metric still reports the
   co-activation structure, which is what you want for the synthetic input (it has no
   decoder). The decoder loader tries `decoder.weight`, `W_dec`, `decoder`, `dec.weight`
   and prints the available keys if none match.

## Untested

The decoder loader has been exercised against `sae_model.pt` from this repo's
`outputs_layerwise/` format only. **It has not been run against the `ae.pt` that
InterPLM's trainer writes.** Run one cell with `--metric geometry --sae <ae.pt>` first and
check that `rho_cooccur_vs_decoder_cos` appears rather than the `decoder not found`
warning — that is the two-minute check before committing the sweep. Same discipline as
`smoke` before `grid` in the InterPLM attack.
