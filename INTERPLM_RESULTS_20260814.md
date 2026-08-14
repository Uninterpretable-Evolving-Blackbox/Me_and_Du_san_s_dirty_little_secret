# InterPLM stress-grid results — 2026-08-14

This snapshot contains the completed compact results from the InterPLM published-metric stress grid run on **Ronnie**.

## Completion and validation

- SAE configurations: **81 / 81**
- valid/test Concept-F1 evaluations generated: **162 / 162**
- floor-vs-SAE comparison tables included: **81 / 81**
- grid completion logs: **3 / 3** (`L11`, `L14`, and `L18`)
- archive files: **449**
- archive SHA-256: `7eec4c1f8689a625c29b9f02dda668fb7f4c6618313edbef3ff21f9365dee93a`

No error, traceback, failed, or killed marker was found at final validation, and no experiment process remained active.

## Files published here

- `interplm_results_20260814.tgz` — compact result archive
- `interplm_results_20260814.tgz.sha256` — archive checksum
- `interplm_results_20260814.tgz.contents.txt` — complete archive file listing

The archive includes aggregate metrics (`concept_f1.txt`, `sae_quality.txt`), all 81 floor-vs-SAE CSVs, training/scoring/grid logs, completion markers, a manifest, and per-file checksums.

## Deliberate exclusions

The 162 raw per-configuration `concept_f1_scores.csv` tables total approximately **31.1 GB** and are therefore not committed to Git. They are summarized in `concept_f1.txt` and remain on Ronnie together with model checkpoints and intermediate embeddings under:

`/home/ronnie/interplm_stress/`

## Verify locally

```bash
shasum -a 256 -c interplm_results_20260814.tgz.sha256
tar tzf interplm_results_20260814.tgz | less
```
