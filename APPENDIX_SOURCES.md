# Building the appendices: what ran, and where every number lives

Written 2026-09-04 for whoever writes the appendices. The main text cites twelve
appendices and none exist yet. This file says, for each one, what it has to
contain, which file the numbers come from, and whether that file is in hand.

`Appendix~D` in the main text is **not ours** — it is a page reference into
Cheng et al. (2024). Do not write one.

---

## The two deliveries

Everything below lives in one of two batches. Neither is in git: the repo
`.gitignore` excludes every output directory, so these arrive as archives.

| Batch | How it arrived | Contains |
|---|---|---|
| **2026-09-02** | `presubmission_results_20260902.zip`, WeChat | depth grid (9 depths), tier-1 denominators, block shuffle, block-18 probes, d_struct at 3 seeds, d_struct untrained |
| **2026-09-03** | **PR #3** on GitHub, unmerged | native fold-disjoint cells, the BH batch, and all fifteen one-copy scripts |

To get a single tree with everything, unpack both into one directory and run

```bash
python verify_paper_claims.py --results <that directory>
```

It should report **22 pass, 2 changed, 0 missing**. If it does not, stop: the
delivery is incomplete and the numbers below will not reproduce. The 2 changed
are the block-18 probes, which are a new depth rather than a contradiction.

Paths below are relative to that unpacked root. `<arm>` is one of
`ckpt_mlm_s{42,43,44}_token` / `ckpt_clm_s{42,43,44}`.

---

## What is in the delivered trees

| Tree | Arms | Depths | Feeds |
|---|---|---|---|
| `outputs_ctrl/ckpt_*` | 6 | 0,4,7,11,14,18,22,26,29 | §4.1, Table 2, App. C |
| `outputs_ctrl/ckpt_mlm_s*_pred` | 3 | all nine | §3 prediction-matched protocol |
| `outputs_ctrl/ctrl_*_ptn` | 2 | all nine | App. G baseline |
| `outputs_ctrl/ctrl_*_ptn_{frozen_decoder,frozen_encoder,soft_frozen_decoder}` | 2 each | 7,14,22 | **App. G** |
| `outputs_ctrl_shuf/ckpt_*` | 6 | all nine | §4.1 |
| `outputs_ctrl_blk16/ckpt_*` | 6 | 11,14,18 | §5 second destruction |
| `outputs_ctrl_folddisj/ckpt_*` | 6 | 11,14,18 | **App. M** |
| `results_interplm_metric/*.csv` | — | 11,14,18 | §4.6, App. P, Table 5 |
| `results_ctrl_saefree_L18{,_mlp}/saefree_by_arm.csv` | 6 | 18 | §4.3, App. E |
| `tier1_results_20260903/logs/s5_bh_*.log` | — | all nine | **BH + omnibus** |

Every cell is `struct_seq_metrics.csv` with columns `feature_idx`,
`{seq,struct}_effect_{obs,shuffle}`, `{seq,struct}_delta`. `struct_delta` is
L_struct. Take the mean over features, then over seeds — in that order. Reversing
it changes the number.

---

## Appendix by appendix

### A, B — full configuration · **write from code**
`train_ctrl_plm.py` (lines 102-139) and `train_sae.py` for the dictionary. The
Reproducibility statement already states most of it; A and B are the exhaustive
version. `run_queue_0804.sh`, the actual launcher, arrived in PR #3 — use it, it
is authoritative over any reconstruction.

### C — scaling-arm values · **MISSING**
§4.1 quotes −4% to +23% order-destroyed and up to 116% native at 500 tok/param.
`run_500tpp.sh`, `run_ppl500.sh` and `run_shuf500_eval.sh` came in PR #3, but the
**outputs did not**. n=1 (seed 42). Ask for them, or drop the appendix and keep
the range in the main text.

### E — probe architectures · **write from code**
`eval_ctrl_saefree.py`. Delivered results for block 18 only:
`results_ctrl_saefree_L18/saefree_by_arm.csv` and the `_mlp` variant, six rows
each with `probe_{helix,strand,burial}_{f1,auroc}` and `contact_p_at_L5`.
Blocks 7/11/14 — the ones Table 3 reports — are **not delivered**.

### F — the 48-row cutoff sweep · **MISSING**
`experiment_contact_def_sweep.py` is committed; no output is in either batch.
Regenerate with `RUN_MUSTRUNS.sh` stage 9 or ask for `results_nshuffle_sensitivity/`.
This is also Figure 2's data, which is why Figure 2 could not be drawn.

### G — frozen dictionary and crosscoder · **IN HAND**
Method from `sae_variants.py` and `crosscoder.py`. Numbers from
`outputs_ctrl/ctrl_{mlm_token,clm}_ptn[_variant]/layer_{7,14,22}`.

Recomputed 2026-09-04, masked minus causal:

| | block 7 | block 14 | block 22 |
|---|---:|---:|---:|
| baseline (`_ptn`) | +0.0146 | +0.0166 | +0.0073 |
| frozen decoder | +0.0210 | +0.0205 | +0.0057 |
| frozen encoder | +0.0218 | +0.0241 | +0.0063 |
| soft-frozen | +0.0196 | +0.0205 | +0.0037 |

All nine variant cells positive, as §3 claims. **Note for the writer:** §3's
"+0.0146 → +0.0217" arrows quote the **frozen encoder** column without saying so.
Either name the variant in §3 or report all three.

### H — the selectivity score · **write from code, numbers in hand**
Definition in `experiment_aa_selectivity.py` (`S = 1 - H(p)/log 20`,
background-corrected). Delivered: `top1_agreement_shuf.csv` in the 09-02 batch,
giving ρ(selectivity, L_struct) = −0.0444 masked / −0.1219 causal, matching §3.
**Open question:** the filename says `_shuf` but the values match §3's native
numbers. Confirm with Ronnie which tree produced it before the appendix asserts
either.

### I — the 8 biochemical groupings · **write from code, complete**
`make_synthetic_layer.py` lines 46-56, verbatim:

| group | residues | group | residues |
|---|---|---|---|
| hydrophobic | AVLIMFWC | small | AGSTCVPND |
| charged | DEKR | tiny | AGS |
| polar | STNQYH | negative | DE |
| aromatic | FWYH | positive | KR |

Plus the 20 single-residue indicators and one all-ones column = 29.

### J — concept-F1 decomposition and Table 5 per-cell · **PARTIAL**
Concept set reproduces exactly from committed inputs: 80 labels = 74 SCOPe
taxonomy + 6 residue-level, at `experiment_concept_f1.py --min-domains 10` over
`cache/scope_40.fa` and `cache/residue_features.csv`. The **per-depth split** and
the 0.039–0.058 test-F1 range are still v1 numbers with no delivered source.
Table 5's other rows: d_struct cells are in `results_interplm_metric/`;
single-latent F1, co-activation ρ and SAEBench are **not delivered**.

### K — the directional split · **method in hand, number is not**
`cpu_stage.py` lines 586-664 documents the up/downstream contact split and warns
that within-model up-vs-down is degree-confounded — report the interaction, not
the raw halves. `analyze_directional.py` computes it. The −0.1422 has **no
delivered source**; regenerate with `run_ctrl_mechanism.sh`.

### M — the fold-disjoint split · **IN HAND**
Construction from `make_folddisjoint_eval_set.py`. Numbers from
`outputs_ctrl_folddisj/<arm>/layer_{11,14,18}`, delivered in PR #3.

Recomputed 2026-09-04:

| block | masked | causal | Δ |
|---|---:|---:|---:|
| 11 | +0.01871 | +0.00167 | +0.01705 |
| 14 | +0.01670 | +0.00147 | +0.01523 |
| 18 | +0.01419 | +0.00417 | +0.01002 |

Mean Δ = **+0.01410**, against +0.01415 on the original split. §2.4's claim
verifies exactly.

**Say the three senses of "held out" here.** (i) L_struct is scored on all 1,500
domains including the 1,350 the dictionary was fitted on; (ii) the fold-disjoint
runs change which domains the dictionary is *fitted* on but still score all
1,500; (iii) only the `_val` half of `outputs_robustness/compute_h1_bootstrap.py`
is both fitted and scored disjointly. No delivered result is both fold-disjoint
and held-out-scored.

### P — reproduction notes for the two reimplemented statistics · **IN HAND**
d_struct: `experiment_interplm_metric.py` (`--gate-mode global|raw`) plus
`experiment_interplm_metric_dsgate.py` and `patch_dsgate.py`, which arrived in
PR #3 and are the per-protein gate that produced Table 3. Concept-F1:
`interplm_attack/` installs and runs their released code; `aa_floor.py` is the
no-model cell. Per-cell d_struct values in `results_interplm_metric/`, 120 cells
across 3 seeds × 3 depths × 2 gates × {native, shuffled} plus 72 random-init.

---

## Numbers that changed, and must not be copied from the current PDF

The delivered data moves nine claims. Do not build an appendix that agrees with
the current `paper/main.pdf` on any of these.

| main.tex | delivered data |
|---|---|
| §4.1 "18 of 18" | **52/54** across nine depths; falls only at `mlm_s44` blocks 26 and 29 |
| §4.3 "27 of 27" (×2) | **3/9** at block 18 — helix and burial reverse in all three seeds; MLP 2/9 |
| §3 "seven of nine survive BH" | **eight of nine**; only 0% fails |
| §5 "denominator is untested" | **18/18** rise under sd, fixed, iqr and rank |
| §5 "One destruction procedure" | block shuffle ran: **14/18**, causal 9/9, masked 5/9 |
| §5 "no untrained baseline was run" | run; passes at both gates and both arms |
| §5 "blocks 11, 14 and 18" | nine depths |
| Table 5 "no verdict — not run" | trained +0.5863 vs untrained +0.2630 (causal, global gate) |
| §4.6 "a single seed" | three seeds |

The BH batch also produced an **omnibus** the paper does not report: pooled across
depths, mean d = +0.2413, 95% CI [+0.1927, +0.2913]; on the held-out half,
+0.1587 [+0.1108, +0.2069]. That is the pre-registered global test a reviewer asks
for when they see nine per-depth CIs with no multiplicity control. It belongs in
the paper, not only in an appendix.

---

## Two things to check before writing

1. **The BH point estimate at 25% depth lies above its own CI**: d = +0.3222,
   CI [+0.1978, +0.2979]. Eight of nine rows are consistent. Percentile bootstrap
   CIs can exclude a point estimate under skew, but a gap of 0.024 at one depth
   only should be understood before it is printed. `fig_bh_depths` marks it.
2. **The selectivity CSV's provenance** — see Appendix H above.

## Figures

`make_figures.py --results <unpacked root> --out paper/figures` draws four, from
the CSVs rather than from typed numbers. `fig_depth_profile` is the strongest
candidate for Figure 1. Figure 2 (the cutoff sweep) cannot be drawn until
Appendix F's data exists.
