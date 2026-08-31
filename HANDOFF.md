# Handoff: where this project stands

Written 2026-08-31 for a fresh agent picking this up on a new local machine.
Read this before touching `PAPER.md`. Everything here is checkable against the
repo or against the result archives; where something is an inference rather than
a fact, it says so.

---

## 1. What the paper claims

Target: ICLR 2027 main track, deadline **2026-09-24**. Draft is `PAPER.md`.

The claim is a negative existence result about measurement practice, not a claim
about protein language models. An SAE-based interpretability metric that scores a
*relation* (do a feature's residues sit near each other in 3D?) can **go up** when
the relation it names is removed from the training data. The paper builds such a
metric (`L_struct`), passes it through seven confound checks, then breaks it with
three attacks:

1. **Corpus control** — retrain on a corpus with residue order destroyed. `L_struct`
   rises in 18 of 18 cells.
2. **No-model layer** — 29 residue-identity indicator functions, no model, no SAE.
   The cysteine indicator outscores every learned feature in 5 of 6 cells.
3. **Random-init baseline** — an untrained network sits at ~+0.018, at or above the
   trained masked arm.

Plus: direct probes favour the *opposite* arm 27/27, and InterPLM's own `d_struct`,
reimplemented, fails the corpus control on the causal arm.

The specimen is a controlled masked-vs-causal pair: one 42M backbone instantiated
twice, sharing initialisation, corpus and batch order, differing only in the
attention mask and the loss built from it.

---

## 2. What was lost, and what it cost

A local drive died. The only thing on it that existed nowhere else was the paper
draft; it has been reconstructed into `PAPER.md` from a pasted copy of the text.

**The repo never held results.** `.gitignore` excludes `outputs_ctrl*/`, `*.pt`,
`*.npy` and every `results_*` directory. So no number in the paper was ever in git.
The numbers live in exactly two places: Ronnie's machine (the GPU box, where all
training and scoring happens) and the result archives he sends.

What *is* committed and does matter: the evaluation set
(`eval_set/{uids,sequences,META}.json`), the residue labels
(`cache/residue_features.csv`) and the SCOPe FASTA (`cache/scope_40.fa`). Those
are the hard-to-recreate inputs. PDB structures are refetchable
(`fetch_pdbs.py`); both corpora regenerate deterministically
(`prep_controlled_corpus.py`, fixed HF stream seed 42).

---

## 3. Who has what

| Location | Holds |
|---|---|
| This repo | The pipeline, the eval set, the labels, `PAPER.md`. No results, no weights. |
| Ronnie's box | All checkpoints, both corpora, every output directory, and ~28 untracked scripts (§5). |
| Result archives | Per-cell `struct_seq_metrics.csv` for whatever batch they cover. |

The most recent archive (2026-08-28/29, branch `runs/rescore-batch`, commit
`f216b2a`) contains 48 cells in three stages. It was sent twice under two names;
the second zip is byte-identical to the first.

---

## 4. Results state

### Already run and in hand

- Main arm effect at nine depths (blocks 0, 4, 7, 11, 14, 18, 22, 26, 29), 3 seeds.
- Corpus control at blocks **11/14/18 only**, 3 seeds, 18/18 cells rising.
- Seven confound checks — but on three different depth grids: corpus control at
  11/14/18, frozen dictionary and crosscoder at 7/14/22, probes at 7/11/14.
- Fold-disjoint refit, native, 18 cells (Δ = +0.01410 against +0.01415).
- From the 08-28 batch: evaluation-distribution control (18 cells), untrained
  seeds 43 and 44 (12 cells), fold-disjoint refit of the *shuffled* arm (18 cells).

### The 08-28 batch, interpreted

**Untrained floor, now n=3** (paper's seed 42 plus new 43/44). Retires the
"single initialisation" limitation. Trained ÷ untrained barely moves:

| Arm | Block | Untrained (n=3) | Trained ÷ untrained |
|---|---|---|---:|
| Masked | 11 | 0.01821 ± 0.00078 | 103.3% |
| Masked | 14 | 0.01778 ± 0.00014 | 93.7% |
| Masked | 18 | 0.01765 ± 0.00065 | 80.3% |
| Causal | 11 | 0.01790 ± 0.00005 | 9.0% |
| Causal | 14 | 0.01846 ± 0.00052 | 7.8% |
| Causal | 18 | 0.01901 ± 0.00009 | 21.7% |

**Fold-disjoint shuffled arm** lands within 1–4% of the original-split shuffled
values at every cell, so the corpus control is not a leak artefact.

**Evaluation-distribution control** splits by arm. Causal keeps essentially all its
elevation when scored on permuted input (−9% at L11, +3% at L18); masked loses
two-thirds and falls *below* the natively-trained masked arm.

### Not yet run

- Corpus control at the other six depths — **`RUN_DEPTH_GRID.sh` stage 1, ~75 min**.
- Probes at block 18 — **`RUN_DEPTH_GRID.sh` stage 2**.
- `d_struct` untrained baseline; fixed-denominator rescoring; a second destruction
  procedure (block shuffle); extra seeds in the 500 tok/param scaling arm (that one
  is ~200 GPU-h and is on hold — see `RUN_500TPP_SEEDS.md`).

---

## 5. What is missing to reproduce

Three separate gaps. Only the first is urgent.

**a. Code that made published numbers but was never committed.** From the archive's
`git_status.txt`, these are untracked on Ronnie's box:

- `experiment_interplm_metric_dsgate.py`, `patch_dsgate.py` — one of §4.5's two gate
  readings. The committed `experiment_interplm_metric.py` offers `--gate-mode
  global|raw`; Table 3's per-protein column corresponds to the untracked variant.
- `patch_raw_coact{,2,3,4}.py`, and a `experiment_raw_coactivation.py.orig` on his
  disk — meaning **the committed co-activation metric is not the version that ran**
  (Table 4, Li et al. row).
- `run_queue_0804.sh` — the training launcher. Recoverable from the Reproducibility
  statement's hyperparameters, but not byte-exact.
- `run_500tpp.sh`, `run_ppl500.sh`, `run_shuf500_eval.sh` — the scaling arm.
- Glue: `label_features.py`, `unwrap_sae.py`, `merge_stage5.py`, `extract_esm2.py`,
  `interplm_sae_encode.py`.

**b. Results.** Nothing is in git. The per-cell `struct_seq_metrics.csv` files are
small (48 cells = 9.7 MB; the full grid is maybe 60 MB) and should be committed —
un-ignore that one filename.

**c. A full rebuild from the repo alone** costs roughly 1–2 days of GPU: corpora
~1 h, twelve 42M models ~9 h, all SAE+metric cells at the measured 125 s each ~10 h,
plus probes and metric experiments. The scaling arm adds ~100 GPU-h on top.

---

## 6. Limitations — the paper's own, plus what was found since

### Stated in §5

- One destruction procedure. A real replication needs a structurally different one
  (block shuffle preserving local windows, or a cross-sequence shuffle).
- Coverage: corpus control at 3 of 30 blocks. **`RUN_DEPTH_GRID.sh` fixes this.**
- The `SD(a_f)` denominator is untested — no fixed or rank-based variant run.
- Dictionary quality differs between conditions: 7 of 30 order-destroyed cells carry
  a degeneracy flag, and order-destroyed dictionaries reconstruct *better* than
  native ones, so a dictionary-quality account of the rise is still open.
- `d_struct` control is a single seed with no untrained baseline.
- Masked-arm pseudo-perplexities are provisional (a mask-token identification was
  corrected during analysis).
- One scale, one modality.

### Retired by the 08-28 batch

- "The untrained baseline is a single initialisation" — now n=3, all 12 new cells
  inside the seed-42 band.
- "The corpus control is computed on the original split" — the shuffled fold-disjoint
  refit exists. **But see the blocker in §7.**
- "Training- and evaluation-distribution shift are not separated" — run, with a
  caveat below.

### Found since, not yet in the paper

1. **Three different things are being called "held out".** (i) `L_struct` is scored on
   all 1,500 domains including the 1,350 the dictionary was fitted on — `cpu_stage.py`
   has no val-restriction flag. (ii) The fold-disjoint runs change which domains the
   dictionary is *fitted* on; they still score all 1,500 (the log says `eval proteins:
   1500 (val held-out: 151)`). (iii) The only genuinely held-out-scored number is the
   `_val` half of `outputs_robustness/compute_h1_bootstrap.py` (d = +0.1797), and that
   is on the fold-*leaky* split. **No result anywhere is both fold-disjoint and scored
   on the held-out set only.** §2.4 currently reads as though the fold-disjoint refit
   was held-out-scored. It should state the three-way distinction, and §4 should say
   plainly that its contrasts share the same eval set on both sides, so contamination
   cannot generate the direction of the result.
2. **The evaluation-distribution control may not be the control the paper describes.**
   Evidence says the activations were left in permuted-index order rather than mapped
   back to original positions: `seq_delta` is unchanged between that run and the
   native-input run in all 18 cells (CLM s42 L11: +0.5296 vs +0.5283), which an
   inverse permutation would have collapsed. If so, the run measures something
   stronger and different — co-activation over a real contact graph whose residue
   identities have been decoupled — and the causal arm holding +0.045 there belongs
   next to the degree-preserving graph null in §4.1, not in §5's limitations.
   **Unverified: the generating script is untracked and commit `f216b2a` is not in
   this clone.** Ask before writing it up either way.
3. **§4.5's description of `d_struct` is inaccurate.** It says they compare "a
   feature's mean activation over residues within 6 Å". Their neighbourhood is
   anchored on a **single** residue per protein (the argmax position), not on every
   active residue — your own `experiment_interplm_metric.py` docstring lists this as
   difference #1. Also the Methods say "up to 100 proteins per feature", not 100.
4. **The "two dashboard files" claim is wrong at current HEAD.** Verified against
   `ElanaPearl/interplm` at `5f4cbf9` (2025-10-30): three dashboard files mention the
   statistic (`feature_activation_vis.py`, `app.py:669-671`, `help_notes.py:5,11`),
   and only one touches the columns. The substance holds — **nothing in the repository
   computes it**; the only per-feature stats the pipeline writes are frequency counts
   (`scripts/collect_feature_activations.py:98-110`). Reword to "no file in the
   released repository computes it".
5. **The gate ambiguity is partly resolvable, in the paper's favour.**
   `interplm/sae/normalize.py:107-131` computes a per-feature maximum **over the
   dataset**, stores it as `activation_rescale_factor`, and every `Dictionary.encode`
   divides by it. That is the convention under which a fixed 0.6 gate is scale-free —
   so the dataset-wide reading is the one their own code supports, and it is also the
   gate where the causal arm fails most cleanly.
6. **Table 4's concept-F1 row may mix two pipelines.** The no-model cell says "4 of 12
   concepts", but InterPLM's Swiss-Prot concept set is far larger than 12; twelve looks
   like the filtered SCOPe+DSSP+RSA set from `experiment_concept_f1.py`, an
   InterPLM-*style* reimplementation. The corpus-control cell (the "81 cells" grid)
   comes from `interplm_attack/`, which genuinely clones and pip-installs their
   package. **Check which pipeline produced which cell before claiming "run using
   their released code".**
7. **"concept-F1" names two different quantities** — a fraction of concepts matched
   (§4.5, 0.0090–0.0620 across 81 cells) and a per-concept F1 value (§4.2, "several
   concepts at 1.000"). Give them separate names.
8. **Naming drift.** `experiment_interplm_metric.py`'s docstring calls the default
   gate `normalised`; the code's choices are `global|raw`.

---

## 7. Asks for Ronnie (zero compute)

1. **Send `outputs_ctrl_folddisj/*/*/struct_seq_metrics.csv`** — the *native*
   fold-disjoint cells. They already exist; they were just not in the last archive.
   Without them the fold-disjoint shuffled numbers have nothing to be divided by, and
   §4.1 cannot be restated on that split. **This is the blocker.**
2. **One question:** in the 08-28 permuted-input stage, were the extracted activations
   mapped back to original residue positions, or left in permuted order? See §6.2 —
   it changes what that result means.
3. **`git add` the untracked scripts in §5a and push.** Five of them made published
   numbers and exist in one copy.

---

## 8. Runs queued

`RUN_DEPTH_GRID.sh` — resumable, one log per cell, packages its own archive.

- **Stage 1**, ~75 min: corpus control at blocks 0, 4, 7, 22, 26, 29 (36 shuffled
  cells at the measured 125 s each; native cells at those depths should already
  exist and are skipped). Takes §4.1 from 18 cells to 54 and puts the corpus control
  on the same nine-depth grid as the main effect — which also brings the
  frozen-dictionary depths (7, 22) and the probe depth (18) inside one grid, so the
  "why these three depths?" objection disappears without retraining any shared
  dictionary.
- **Stage 2**: probes at block 18, linear and MLP. Block 18 is the deepest control
  depth and where the masked `L_struct` advantage has already halved, so it is where
  27/27 is most likely to change.

Preflight refuses to run if the real and shuffled checkpoint roots hash to the same
weights — that failure produces plausible numbers ("shuffled ≈ real") rather than an
error, and it would look like the metric passing.

**Expect stage 1 to change a number.** Blocks 0 and 4 are where the arm effect is
near zero and where BH correction already fails; the rise may not hold there. A
depth profile that says "rises at seven of nine depths, flat at the two where there
is no effect to begin with" is stronger than three depths and a suspicion.

**Do not** retrain shared dictionaries or the crosscoder at 11/18 (days, for a
tidier table), and do not start the 500 tok/param seeds (~200 GPU-h, on hold).

---

## 9. Do not do this

- Do not delete depths to make the grids look consistent. The intersection of the
  three grids is block 14 alone, and the frozen-dictionary check does not exist at
  11 or 18 — you cannot report a number that was never computed.
- Do not treat an unrun, undefined or underpowered cell in Table 4 as a failure.
- Do not quote absolute `L_struct` values as exact: the 5-permutation null shifts
  them by up to +0.0187. Between-condition comparisons are unaffected.
