# Controlled MLM-vs-CLM experiment — run status

Last updated: 2026-07-16 (Ronnie/CST)

## Summary

- Repository revision: `c0dff9f` (`main` at setup time)
- Smoke test: **passed end-to-end** and printed `DONE`
- Full seed-42 experiment: **running** on Ronnie
- Token-matched CLM and MLM training: **complete**
- Current stage at last check: automatic layerwise analysis of the trained models
- Observed training throughput: about `233k tokens/s`
- No experiment hyperparameters were changed.

## Runtime environment

Ronnie:

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (about 96 GB VRAM)
- Driver: `595.71.05`
- Python: `3.12.13`, installed in user space with `uv`
- PyTorch: `2.11.0+cu128`
- CUDA reported by PyTorch: `12.8`; `torch.cuda.is_available() == True`
- Architecture anchor: `esm==3.2.3`
- Free disk before setup: about `1.3 TB`

The host's system Python was 3.13 without `pip`/`ensurepip`, so a project-local
Python 3.12 environment was created without `sudo`. The unconstrained 2026
dependency resolver selected an obsolete `numba/llvmlite` combination in order
to accommodate a newer NumPy release. Installation therefore used the minimal
environment constraint `numpy<2`, resolving to NumPy `1.26.4`, `numba 0.66.0`,
and `llvmlite 0.48.0`. Repository files and experimental hyperparameters were
not changed.

## China-network handling

Direct access from Ronnie to `huggingface.co` timed out. `hf-mirror.com` was
reachable and successfully produced a 2,000-sequence test corpus. For the full
corpus, the repository's unmodified preparation script was instead run on the
Mac with direct Hugging Face access, then the generated arrays were transferred
to Ronnie over Tailscale using resumable `rsync`.

All 1,500 required PDB entries were downloaded on Ronnie successfully:

```text
done: ok=1500 skipped=0 failed=0
All eval structures present.
```

## Corpus preparation and the token-count discrepancy

The unmodified command and default parameters were used:

```bash
python prep_controlled_corpus.py
```

Relevant fixed settings were 3,000,000 sequences, context 512, shuffle seed 42,
shuffle buffer 50,000, the bundled SCOPe holdout, and the ESM-C tokenizer.

Generated metadata:

```text
n_sequences = 3,000,000
n_tokens    = 1,056,768,706
mean_len    = 352.3
dropped     = non-std-aa 24,510; too-short 22; SCOPe holdout 29
```

The README says to expect approximately `1056.7M` tokens. The script printed
`1056.8M` because the exact result was `1,056,768,706`, a difference of about
68,706 tokens (0.0065%) from 1056.7M. We paused before training, reported the
difference, and received approval to continue. The 2,000-sequence smoke corpus
produced the same aggregate count whether obtained through the mirror on Ronnie
or direct Hugging Face access on the Mac (`705,320` tokens).

For stronger future reproducibility, the Hugging Face dataset revision should
be pinned explicitly; the current script names `ConvergeBio/uniref50` without a
revision.

## Transfer verification

The full corpus was transferred to:

```text
/home/ronnie/own_sae_data/uniref50_pilot
```

SHA-256 values matched between the Mac and Ronnie:

```text
23662025d5c6cabc6c26aa3ff52323dc83a29d0b97e7777176366a6231a21fda  lengths.npy
a562cc34c7438fe36485686f0b461e204191cf9c19ac9af2e79522bf6ce5890d  meta.json
e09abc2e1643158e8e20dd7e45fbc17b09786a8d9ccac2c20c0190e3b015530c  offsets.npy
91b96bb7fdefee671c6a1091402b2a097c7e04b12f554b1e12542e45f08311ef  tokens.npy
```

Ronnie's `meta.json` was independently read after transfer and reported the
expected 3,000,000 sequences, 1,056,768,706 tokens, context 512,
`ConvergeBio/uniref50`, and the ESM-C tokenizer.

## Smoke test

The repository-prescribed command was run on Ronnie:

```bash
SMOKE=1 bash run_full_ctrl.sh
```

It passed CUDA and ESM anchor checks, trained the small CLM and MLM models,
extracted both tested depths, trained the SAEs, ran the CPU structural analysis,
ran concept-F1, completed the H1 bootstrap, safely pruned consumed `Z.npy`
intermediates, and printed:

```text
########## DONE Thu Jul 16 10:03:09 CST 2026 ##########
```

The repeated Biopython `pairwise2` deprecation messages were warnings only. A
Mac-only `multiprocess.resource_tracker` shutdown exception also appeared after
successful corpus output and did not affect the generated files or hashes.

## Full experiment

The full run was started with the repository defaults and no overrides:

```bash
nohup bash run_full_ctrl.sh > train.log 2>&1 &
```

Startup log:

```text
device: cuda | torch 2.11.0+cu128 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition
esm anchor OK
seeds:42 | protocols:token pred
CLM | 42.0M params | steps 40283 | ~660M tok
```

The driver PID recorded on Ronnie is `90269`. At the latest check it remained
alive. Token-matched CLM completed with reported final train loss `2.7408`, and
token-matched MLM completed with reported final train loss `2.5428`; both final
checkpoints were saved. The pipeline then advanced automatically into layerwise
analysis and was processing `ckpt_clm_s42` layer 4. GPU utilization may be zero
during CPU-heavy analysis stages. No traceback, CUDA warning, failed-stage
marker, or pruning refusal was present, and about 1.2 TB of disk remained free.

`run_full_ctrl.sh` automatically chains the remaining stages:

1. token-matched CLM (shared arm)
2. token-matched MLM
3. prediction-matched MLM
4. layerwise extraction and SAE training
5. structural-locality and concept-F1 analyses
6. H1 bootstrap
7. guarded intermediate pruning
8. final result checklist

Seeds 43 and 44 are intentionally **not** included in this stage; the README
labels them as later replicates to run only if requested.

## Monitoring and stop conditions

The run is monitored for process liveness, GPU utilization, disk space, ETA,
validation/checkpoint progress, tracebacks, CUDA failures, failed stages, and
`REFUSING to prune Z.npy`. Any condition that could affect scientific validity
will stop automatic progression and be reported rather than silently worked
around. When the full run completes, the small result CSV/JSON files and
`train.log` will be packaged as `ctrl_results.tgz` using the README command.
