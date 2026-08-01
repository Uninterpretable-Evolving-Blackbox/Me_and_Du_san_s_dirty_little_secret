# Stage 6 — does `L_struct` reward composition alone?

```bash
git pull
STAGE=6 bash RUN_MUSTRUNS.sh
```

About a minute a cell. No SAE training, no GPU. **Do this before anything else on the list** — it
may change what the other runs are for.

> **Retraction, if you read the previous version of this file.** It said real SAE features denser
> than `topk_frac` "are getting `d = 0` throughout the project, with no warning anywhere." **That is
> false and I withdraw it.** I inferred it from the synthetic run instead of checking the real data.
> In the 33.2M layer-6 cell, 824 of 3,840 real features have occupancy above 0.10 and **not one** has
> `struct_delta` exactly 0; every dense feature's 90th percentile is a genuine interior value
> selecting a proper top decile (~29,370 of 293,760 residues). The degeneracy needs activations
> **tied** at the threshold, which only binary indicators produce. **No number in the paper is
> affected.** Details below under "the one artifact".

---

## Why

`L_struct`'s permutation null shuffles residue *positions* against a **fixed structure**. So any
property with spatial autocorrelation in the fold produces excess structural co-activation without
encoding anything about structure. Hydrophobicity is the obvious case: hydrophobic residues are
disproportionately buried, buried residues have high contact degree, and their long-range partners
are disproportionately hydrophobic too.

The paper's §5.4 rejects this using an amino-acid **selectivity** measure — one minus the normalised
entropy of a feature's activation over the 20 types. That measure is structurally blind to a **class**
detector: a feature firing on all hydrophobic residues is spread over ~8 types and therefore scores
as maximally *un*selective. So the existing test could not have caught this even if it were true.

## What the stage does

Builds synthetic "features" that are pure indicator functions of residue identity — `residue == C`,
`residue ∈ {hydrophobic}` — with **no model, no SAE, no training anywhere** — and pushes them through
the identical metric path: same neighbour graphs, same permutation null, same `--n-shuffles 5`, same
`struct_delta = observed − shuffled`.

If an indicator scores comparably to a real learned feature, the metric rewards composition.

## What it already returned on the laptop

33.2M pair, layer 6, 8 Å / gap ≥ 12:

| feature | `struct_delta` | |
|---|---|---|
| **`class:hydrophobic`** | **+0.1905** | **above the p99 of the real features** |
| `aa:C` (pure cysteine indicator) | +0.4221 | above the *max* of the real features |
| `aa:V` / `aa:I` / `aa:L` | +0.145 / +0.125 / +0.114 | at or above p99 |
| `class:charged` | −0.0212 | |
| `aa:K` / `aa:E` | −0.066 / −0.064 | |

Real SAE features in the **same cell** (n = 3,840), *rescored at the same `topk_frac` so both sides
are under one rule*: p50 +0.0101, p90 +0.0567, p99 +0.1340, **max +0.3898**. (At the project's 0.10
they are p99 +0.1373, max +0.3925 — the real distribution barely moves, so the comparison does not
depend on the choice.)

So `class:hydrophobic` beats **99.6%** of the learned features and `aa:C` beats **100%** of them,
both like-for-like.

**Lead with the `class:hydrophobic` row, not cysteine.** It is the weaker number but the stronger
argument: a class detector is exactly what §5.4's selectivity measure is blind to, so that row is the
one the paper's existing test could not have caught. Cysteine is a single type and §5.4's measure
*would* have scored it as highly selective — it is a striking number but not a hole in the test.

And the ordering is the burial pattern exactly — buried residues positive, surface residues negative.
It is not a rarity artifact: tryptophan is about as rare as cysteine (occupancy 0.0135 vs 0.0116) and
scores +0.0368 against cysteine's +0.4221, and across the 20 single types Spearman(occupancy,
`struct_delta`) = −0.150, p = 0.53.

What we want from your run: does this hold across depths and on both arms, or is layer 6 of the
33.2M pair special?

## The one artifact, and its actual scope

A **binary** feature active on more than `topk_frac` of residues is silently zeroed. The selection is

```python
thresh = np.percentile(acts_chunk, 100*(1 - topk_frac), axis=0)
active = acts_chunk > thresh          # strictly greater
...
d[n_active < 5] = 0.0
```

For a binary feature with occupancy above `topk_frac`, that percentile is exactly 1.0, nothing is
strictly greater, `n_active` is 0, and `d` is forced to 0.0. On the first laptop run all eight class
indicators returned exactly `0.0000` and it nearly got read as a null result.

**This requires values tied at the threshold, so it affects binary indicators only.** Real SAE
activations are continuous — see the retraction at the top. The script now **refuses to emit** a zero
rather than let one read as a null, so you cannot hit this silently.

**The stage runs once, at `--topk-frac 0.60`.** For a binary feature the score is *invariant* to
`topk_frac` provided `topk_frac > occupancy` — verified, max |0.60 − 0.50| = 0.0000000000 across all
28 features, and max |0.60 − 0.10| = 0.0000000000 across the 20 single types. `class:small` has the
largest occupancy at 0.4929, so 0.60 clears everything with a 0.107 margin. **Every row is directly
comparable to the paper's numbers** — an earlier version of this file wrongly warned otherwise.

Where `Z.npy` is present the stage also passes `--compare-real`, which rescores that cell's real
features at the same `topk_frac` in the same run, so both sides are scored under one rule. Without
it the printed real percentiles come from `struct_seq_metrics.csv`, computed at the project default
of 0.10. The stage does **not** otherwise need `Z.npy`; it builds its features from the sequences.

## Reading the output

`results_synthetic_composition/<arm>_L<layer>.csv`, plus a printed summary per cell ranking the
synthetic features and giving the real-feature percentiles beneath them.

The number that matters: **best synthetic `struct_delta` versus the p99 and max of the real features
in the same cell.** If the synthetic is at or above p99, composition alone reaches the top of the
learned distribution.

One caveat on that comparison even with `--compare-real`: the active sets are different sizes.
Cysteine's is 1.16% of residues; a real feature's selection is a top decile. Cohen's *d* is
standardised, so this is not fatal, but it is why the script now prints **best CLASS** on its own
line — the class rows have occupancy in the same range as the real selection and are the fair
comparison. Quote `class:hydrophobic` beating 99.6% as the headline; quote cysteine as the extreme.

## What it would mean

If it replicates, the composition account is **not** refuted and §5.4's rejection does not stand. It
would also explain, in one mechanism, several things currently treated as separate findings: why
shuffled-trained models score higher (they are closer to pure composition detectors), why the
inflation worsens at higher separation floors (the ≥12 graph is more core-biased), and why the causal
arm's real `L_struct` collapses to 0.0011 at mid depth while its shuffled counterpart sits near 0.05.

That is a cleaner and more general result than the one the paper currently claims.

## Where this sits relative to the other stages

Stages 0–4 have run; the results are in `mustruns_20260731.tgz`. Still outstanding, in order:

1. **This stage.** Fast, and may reframe the rest.
2. **`L_struct` on raw activation dimensions.** The metric has *never* been computed on anything but
   SAE features — checked, there is no SAE-free variant anywhere in either repo. If shuffled/real
   inflates on raw dimensions too, the sparse-feature framing is wrong about its own object. Needs a
   `--raw` path in `cpu_stage.py`, which does not exist yet.
3. **Shuffled training at seeds 43/44** (`STAGE=5`). The whole shuffled control is seed 42 only,
   while everything else in the paper is three seeds.

The separation-6 control cell mentioned in an earlier note is **no longer worth running**; we now
know what governs that axis.
