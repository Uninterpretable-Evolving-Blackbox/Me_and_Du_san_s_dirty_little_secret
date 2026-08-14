# The verification queue — run this next, not job 4

```bash
git pull
bash run_checks.sh
```

**About 3 hours if you skip the one GPU stage; four of the five stages need no GPU at all.**
Start it and walk away.

Job 4 (`RUN_500TPP_SEEDS.md`) is **still on hold** and nothing about it has changed: it still
owns the GPU for days and it still only tightens an error bar on a result we already have.
These five checks are what the paper needs next.

Thank you for the completed grid — 81/81 SAEs, 162/162 evaluations, all three layer logs at
`GRID COMPLETE`, and the archive checksum verifies. Routing the tail through the dense path to
reach CUDA was a genuinely good call and it is what got the causal arm finished.

**One request about that, and it is the only thing here that is urgent.** 19 of the 162
cell-splits were scored through the GPU dense path and 143 through the CPU sparse path, and no
cell was scored both ways — so there is no evidence either way about whether the two paths
compute the same number. Comparing across conditions is confounded with model seed, so it does
not settle it. **Could you re-score one cell that already has a GPU result — say
`clm_s44_L18_s0` — on the CPU path and send both numbers?** It is one cell and it either
removes the question or tells us something we need to know before the grid is quotable.

---

## The five stages

### 1 — Grid audit · seconds · reads files, no compute

```bash
python check_grid.py --base ~/interplm_stress
```

Prints cells per condition and runs the comparability gate on `sae_quality.txt`. Free to run
whenever you want a status line.

Already run here on your archive: **81/81 SAEs, 81/81 scored**, all five conditions present
(`clm`, `mlm`, `shuf_clm`, `shuf_mlm`, `untrained`). The gate **fails** — the causal arm holds
1261 live features and L0 176.8 against the masked arm's 973 and 103.5, i.e. 1.30× and 1.71×.
That is now known to be a property of the two objectives rather than an artefact of a partial
grid, which is worth knowing and is itself a result.

### 2 — Larger permutation null on the headline cells · ~1–2 h · CPU

Flagged essential in the supervisor feedback and the last of that set still open.
`results_nshuffle_sensitivity/` has 5-vs-25 already — but only for the **contact-definition
sweep**. Every headline `L_struct` number is still at 5 permutations.

Runs the 18 headline cells (3 seeds × 2 arms × layers 11/14/18) at 25 and diffs them against
the stored 5-permutation values, cell by cell. **A small maximum difference is the good
outcome**: it means the null was already big enough and the published numbers stand.

> `cpu_stage.py` writes `struct_seq_metrics.csv` **into the directory you give it**, so running
> it on the real layer dirs would overwrite the 5-permutation results the paper reports. Each
> cell runs in a scratch directory of **symlinks** instead — nothing is copied, no 1.5 GB `Z.npy`
> duplication, and the originals cannot be written through it.

### 3 — Fold-disjoint SAE/probe split · report free · re-fit ~1–3 h GPU

Concept-F1 already uses a fold-disjoint val/test split; the 1,350/150 split that fits the
**dictionaries and probes** is a uniform random partition. Only one of the two is
homology-aware, and there is no principled reason for that.

**Measured before asking for GPU time: 118 of the 150 held-out domains — 78.7% — share a SCOPe
fold with a domain in the 1,350 used for fitting.** The replacement takes that to 0 of 151. So
it is a real leak, not a cosmetic one.

`eval_ctrl_plm.py` reads the split from `<eval-set>/META.json:val_uids` and nothing else, so
this needs **no code change and no pLM retraining**. The stage **reports by default and re-fits
nothing**; to rebuild:

```bash
FOLDDISJ_APPLY=1 bash run_checks.sh
```

It writes to `outputs_ctrl_folddisj/`, leaving `outputs_ctrl/` untouched.

### 4 — Concept scores against their prevalence floor · ~30 min · CPU

A binary concept score only means something if it beats the classifier that marks every residue
positive. That classifier scores `2p/(p+1)` for a label of prevalence `p` — 0.33 at p=0.20,
0.60 at p=0.42. Helix and burial are prevalent enough that the floor is most of the reportable
range, so a concept number quoted without its floor says very little.

Reports, per cell and for helix / strand / burial: prevalence, floor, the best single feature's
F1 (threshold picked on val, reported on test), and the **margin over the floor** — which is
the only part attributable to the model. It also runs the random-init cells, which should sit
*at* the floor and are the contrast that makes the trained numbers readable.

Label definitions, the `sasa` column and the fold-level split all come from
`experiment_concept_f1.py`, and the floor from `experiment_extra_metrics._prevalence_floor`, so
there is one definition of each rather than a second copy that can drift.

*Measured: 99 s for a 3,840-feature dictionary on a laptop.*

### 5 — The global level: remote-homology probe · ~1 h · CPU

The readout suite was meant to separate three levels — position-wise, pairwise and global.
Position-wise is the linear probes and pairwise is `experiment_pairwise_probe.py`. **Global was
never built.** This is it.

Fold classification is not usable here: 1,500 domains over ~430 folds leaves most folds with a
handful of members, and a fold-disjoint split is impossible when the fold *is* the label. So it
uses the standard remote-homology framing:

- **positive**: same fold, **different superfamily** — homologous but not close
- **negative**: different fold
- **split**: superfamily-disjoint, so no superfamily is on both sides

Everything else mirrors the pairwise probe exactly — same symmetric `|a-b| ‖ a*b` encoding, same
`LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=42)`, same
shuffled-label control drawn from its own generator, same `--seed 42` and `--split-seed 1234`.
The three levels therefore differ only in the relation predicted, not in the estimator.

**Read the shuffled-label control first — it must sit at ~0.5.** On a published-model
dictionary here it gives AUROC 0.616 with the control at 0.508.

> ⚠️ A caveat that cost us a number already. An earlier version of this split grouped each pair
> by its first member only, so a test pair's second domain could sit in a training superfamily.
> That version reported **0.880** against the corrected **0.616** — the leak was worth **+0.264**
> — and **the shuffled-label control did not catch it**, reading 0.504 leaky and 0.508 strict.
> Permuting labels tests for label leakage and is blind to group leakage. The split now requires
> BOTH endpoints of a pair on the same side and reports how many pairs it drops for straddling.

It also runs the raw-activation contrast automatically wherever `outputs_raw_real/<arm>/layer_<L>/X.npy`
exists — the same place the pairwise probe reads it, so `STAGE=10` having been run is the only
precondition. That contrast is what makes "does the dictionary earn its place?" answerable
globally as well as pairwise, and the stage says so loudly rather than skipping quietly if the
raw activations are missing.

*Measured: 268 s for a 3,840-feature dictionary at `--n-pairs 5000` on a laptop.*

---

## Controls

```bash
ONLY=4 bash run_checks.sh             # one stage
SKIP="3" bash run_checks.sh           # everything except the GPU stage
FOLDDISJ_APPLY=1 bash run_checks.sh   # stage 3 actually re-fits
```

Idempotent and skip-if-done throughout; a failing stage logs and moves on rather than idling
the box.

## Send back

- `logs_checks/` — carries the git revision, which is how we know which code produced which number
- `results_nshuffle_headline/` (2) · `eval_set_folddisj/META.json` (3)
- `results_trivial_baseline/` (4) · `results_global_probe/` (5)
- `outputs_ctrl_folddisj/*/*/struct_seq_metrics.csv` — only if you ran `FOLDDISJ_APPLY=1`
- the one re-scored cell from the GPU/CPU comparison above

## If a stage dies immediately

Stages 2, 4 and 5 need `Z.npy` in each layer dir and skip per cell with a message if it was
pruned, rather than producing an empty comparison. Stages 3 and 5 need `cache/scope_40.fa`;
stage 4 needs `cache/residue_features.csv` with `ss_8class` and `sasa`, and **stops with an
error rather than reporting a zero** if `ss_8class` turns out to be unfilled. Stage 4 also
wants the random-init cells under `outputs_ctrl_randominit/` (`STAGE=15`) and stage 5 wants the
raw activations under `outputs_raw_real/` (`STAGE=10`); both say so rather than skipping
quietly, because those two are the contrasts that make the trained numbers readable.
Everything else soft-fails — send the last 50 lines plus `logs_checks/`.
