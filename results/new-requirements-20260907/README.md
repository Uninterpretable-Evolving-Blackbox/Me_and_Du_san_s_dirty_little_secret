# New experiment requirements — 2026-09-07

Code revision run on Ronnie: `643c48ae70a6b244ac114166a147a06e73d20f50`

The latest `README.md`/`START_HERE.md` queue was executed in the prescribed
order. All five commands completed successfully or found their outputs already
cached; no new model training was needed in this batch.

## Queue status

| Command | Status | Delivered output |
|---|---:|---|
| `STAGE=13 bash RUN_MUSTRUNS.sh` | Complete/cached | 2 files, 16 sequence-gap values × 3 layers for real and shuffled roots (Appendix F/Figure 2) |
| `STAGE=1 bash run_ctrl_mechanism.sh` | Complete | `results_rigor/directional_mechanism.csv`, 9/9 cells |
| `STAGE=10 bash RUN_MUSTRUNS.sh` | Complete/cached | 4 raw-dimension L_struct/perplexity CSVs; raw activations were already present for Stage 11 |
| `STAGE=11 bash RUN_MUSTRUNS.sh` | Complete/cached | 6 SAE probes + 6 raw probes + ESM-2 positive control |
| `NSHUF_HI=25 ONLY=2 bash run_checks.sh` | Complete/cached | 18/18 headline cells |

The two prerequisite paths requested by the queue were also checked:

- `~/own_sae_data/uniref50_pilot_shuf_500tpp` exists.
- `outputs_layerwise/esm2/layer_16/raw_embeddings.npy` exists.

## Main results

### Directional mechanism test (Appendix K)

- Structural interaction (downstream − upstream): **mean −0.1422**, positive in
  **0/9** cells; across-seed `t=-15.46`, `p=0.004` (`df=2`).
- Sequential control interaction: mean `+0.0369`, positive in 7/9 cells;
  `p=0.166`.
- Net interaction (structural − sequential): **mean −0.1791**, positive in
  **1/9** cells; `p=0.008`.

The script explicitly classifies this as **mixed/inconclusive** for the proposed
bidirectionality mechanism. Do not reinterpret the raw upstream/downstream
halves within one arm; they are on different degree scales.

### Raw-dimension control and pairwise probes

Stage 10’s official summary reports shuffled-minus-real raw L_struct values of
`+0.01890`, `+0.02942`, `+0.02733` for causal layers 11/14/18, and
`−0.00067`, `+0.00230`, `−0.00010` for masked layers 11/14/18. The accompanying
perplexities are labelled separately because masked values are pseudo-
perplexities and are not comparable to causal values.

Stage 11’s ESM-2 positive-control AUROC is **0.6443** (shuffled-label control
`0.4884`), confirming that the probe has power. The controlled raw and SAE
AUROCs are in `results_pairwise_probe/` and the official summary.

### Permutation-count stability

The 25-permutation check covers all 18 cells. Relative to five permutations,
the maximum absolute change is **0.00023** and the median is **0.00007**.

### Appendix C scaling data

The 500-token/parameter shuffled directory exists, so no 50-GPU-hour training
was started. Ronnie already has **24 seed-42 real/shuffled metric files**
(six model/layer cells per root, each with `META.json` and
`struct_seq_metrics.csv`); they are included here for appendix preparation.

## Verification

- `tests/run_all.sh`: all suites passed.
- `preflight.sh all`: 12 passed, 0 failed, 3 skipped.
- `logs_newreq_20260907_005004.log` records the exact queue and revision.
- `reviewer_batch_summary_20260907.txt` is the repository’s official summary
  output.

The compact archive and its SHA-256 checksum are included. Raw activation
arrays are intentionally not copied into this GitHub result bundle because
they are multi-gigabyte intermediates; the machine-readable metrics and logs
are complete.
