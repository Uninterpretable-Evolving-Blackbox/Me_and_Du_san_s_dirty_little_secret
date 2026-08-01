# Stage 6 — does `L_struct` reward composition alone?

```bash
git pull
STAGE=6 bash RUN_MUSTRUNS.sh
```

About a minute a cell. No SAE training, no GPU. **Do this before anything else on the list** — it
may change what the other runs are for.

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
| `aa:C` (pure cysteine indicator) | **+0.4221** | |
| `class:hydrophobic` | +0.1905 | |
| `aa:V` / `aa:I` / `aa:L` | +0.145 / +0.125 / +0.114 | above the p99 of real features |
| `class:charged` | −0.0212 | |
| `aa:K` / `aa:E` | −0.066 / −0.064 | |

Real SAE features in the **same cell** (n = 3,840): p50 +0.011, p90 +0.060, p99 +0.137, **max
+0.3925**.

**An indicator function beats 100% of the learned features.** And the ordering is the burial pattern
exactly — buried residues positive, surface residues negative.

What we want from your run: does this hold across depths and on both arms, or is layer 6 of the
33.2M pair special?

## One thing to know before reading any output

**Any feature active on more than `topk_frac` of residues is silently zeroed.** The selection is

```python
thresh = np.percentile(acts_chunk, 100*(1 - topk_frac), axis=0)
active = acts_chunk > thresh          # strictly greater
...
d[n_active < 5] = 0.0
```

For a binary feature with occupancy above 10%, the 90th percentile is exactly 1.0, nothing is
strictly greater, `n_active` is 0, and `d` is forced to 0.0. On the first laptop run all eight class
indicators returned exactly `0.0000` and it nearly got reported as a null result. It is an artifact.

That is why the stage runs **twice per cell**:

- `--topk-frac 0.10` — the project setting. Valid for the 20 single-type indicators, all of which
  have occupancy under 10%. **Ignore the class rows in these files.**
- `--topk-frac 0.50` — above every class occupancy, so the class indicators are measurable. Not the
  project setting, so do not compare these numbers against the paper's.

This also has an implication beyond this stage, and it is worth checking separately: real SAE
features denser than `topk_frac` are getting `d = 0` throughout the project, with no warning
anywhere.

## Reading the output

`results_synthetic_composition/<arm>_L<layer>_tk<frac>.csv`, plus a printed summary per cell ranking
the synthetic features and giving the real-feature percentiles beneath them.

The number that matters: **best synthetic `struct_delta` versus the p99 and max of the real features
in the same cell.** If the synthetic is at or above p99, composition alone reaches the top of the
learned distribution.

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
