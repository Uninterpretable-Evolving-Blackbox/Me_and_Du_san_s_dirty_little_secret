# InterPLM's own published metric, run on our models

```bash
git pull
./preflight.sh env && ./preflight.sh ckpt   # STOP on any FAIL
cd interplm_attack
export CKPT_ROOT=$HOME/own_sae_data/uniref50_pilot
export CKPT_ROOT_SHUF=$HOME/own_sae_data/uniref50_pilot_shuf   # SEPARATE tree; read from your METAs
./RUN_INTERPLM_STRESS.sh setup     # ~2-3 h, once
cd .. && ./preflight.sh post-setup && cd interplm_attack   # verifies the generated trainer
./RUN_INTERPLM_STRESS.sh smoke     # 2 min — please don't skip
./RUN_INTERPLM_STRESS.sh grid      # 9 models x 3 layers x 3 SAE seeds
# ...then, once the first cell finishes and BEFORE leaving it running:
cd .. && ./preflight.sh first-cell
```

Send back: `$BASE/results/` (default `$BASE=$HOME/interplm_stress`).
`SAE_SRC` defaults to this repo root, where `model_ctrl_esmc.py` already lives.

## How long

**Estimate, ~1–2 days serial. Only one component of it is measured**, so treat it as a
planning figure and not a promise:

| stage | count | each | basis |
|---|---|---|---|
| `setup` | once | 2–3 h | measured |
| embedding | 54 passes (9 models × 3 layers × {analysis, train}) | minutes | GPU, forward only |
| SAE training | 81 | ~10–20 min | **estimated** |
| `compare_activations` | 648 (81 cells × 8 shards) | **~6 min/shard at 1,280 features** | measured 2026-08-07, see below |
| `calculate_f1`, `report_metrics`, `aa_floor` | 81 each | seconds | — |

**CORRECTION (2026-08-07).** An earlier version of this file said `compare_activations`
was "~5 min" per split and put the floor at ~13 h. That was wrong by roughly 5x — the
5-minute figure came from a much smaller configuration. Measured directly while running
the same code on RITA_l over the same annotation set:

```
Calculating over 12288 features in 50 chunks   ->  ~72 s/chunk  =  60 min per shard
                                                   =  0.293 s per feature per shard
```

The cost is linear in **features x shards**, so at `CTRL_EXP=4` (1,280 features):

```
1,280 x 0.293 s = ~6.2 min/shard  ->  ~50 min/cell (8 shards)  ->  ~68 h for 81 cells
```

So the grid is **~3 days serial on hardware like ours**, not 1-2. It is single-threaded,
so `xargs -P 3` over cells brings that to roughly a day — but size N against RAM, not
cores (each peaks at 12-15 GB; 6-way turned a 5-minute job into 187 minutes).

That 0.293 s/feature/shard was measured on an Apple M-series CPU. Rescale it on your own
box from the first cell rather than trusting it: watch the `n/50` chunk pacing in the
first `compare_activations` and multiply out before committing to the full grid. **If it
comes out far above 3 days, stop and tell me** — the honest options are then fewer seeds
(3 -> 1, cutting it threefold) or fewer layers, and that is a call to make before burning
the time, not after.

**It resumes, but only because guards were added on 2026-08-07 — an earlier version of
this file claimed resumability it did not have.** The `.done` sentinels cover the
embeddings only, and those get deleted per layer, sentinel included. SAE training and
scoring now have their own guards: a cell is skipped if `models/grid/<tag>/ae.pt` exists
(their trainer writes it last, so its presence means finished) or if
`results/grid/<tag>_<split>/concept_f1_scores.csv` is non-empty. You will see
`skip (SAE exists)` / `skip (scored)` lines on a restart. **This resume path has not been
exercised** — if you do restart, eyeball that the skip counts look right before letting it
run on.

Stopping early is fine. Partial cells are usable; send what exists.

**What persists:** the SAEs, under `models/grid/<tag>`. Only the embeddings are deleted
per model (`rm -rf` at the end of each layer), which caps disk at ~60 GB instead of
~210 GB. So adding another metric later costs a re-embed, not a re-train — the cheap half.

**This is new and unrelated to the Aug 6/7 batch.** It does not use `RUN_MUSTRUNS.sh`
and does not touch anything in it — it builds its own venv with InterPLM installed,
because the whole point is to run *their* code rather than ours.

---

## What it is

Every metric module here is InterPLM's own (Simon & Zou, *Nature Methods* 2025),
invoked unmodified as `python -m interplm.analysis.concepts.*`. The only substitution
is the backbone: their metric, our models. `embed_ctrl_interplm.py` is the single piece
of our code in the path, and it exists only to write our activations in the format
their `embed_annotations.py` produces.

## Why it is worth your GPU

Every published pLM SAE paper — InterPLM, Adams et al., the Matryoshka ESM2-3B work,
Villegas Garcia & Ansuini, Nainani et al. — evaluates on **ESM-2**. Not one uses a
causal pLM. The field therefore carries an untested assumption: that these metrics
behave the same regardless of training objective. We have the only objective-isolated
pair, a shuffled-corpus arm and three seeds, so this is the only place it can be tested.

**Your Aug 6/7 result is what makes it sharp.** You showed the validity failure is
regime-bounded: invalid for causal at any budget and *worsening* with training
(26× → 75× at L11), invalid for masked only when undertrained, roughly valid for
well-trained masked (0.90× at 500 tok/param, shuffled below real). And ESM-2's own
features are fold detectors — 2.3% amino-acid-pure against our 42M masked arm's 63–85%.

The obvious next question is whether **InterPLM's published metric shows that same
regime structure on the same models**. If it does, the finding generalises beyond our
statistic. If it doesn't, we learn the failure is specific to `L_struct` — which is
equally worth knowing and considerably narrows the paper.

## Four things that will bite

1. **`CKPT_ROOT_SHUF` must be a different tree from `CKPT_ROOT`.** If the shuffled arms
   resolve to the real checkpoints, the control compares the real models against
   themselves and reports "shuffled == real" — which reads as the metric *passing* its
   validity check. A sha256 preflight now refuses to start if any two arms hash the
   same, so it catches copies and symlinks, not just equal paths. You should see
   `preflight OK: 8 checkpoints, all distinct`.

2. **`extract_annotations` will eat the box if you let it.** It defaults to
   `ProcessPoolExecutor(os.cpu_count())`, each worker holding a fully exploded
   per-residue DataFrame at 15–30 GB. It took a 128 GB laptop down twice. `setup` pins
   `--max_workers 1 --n_shards 16`; both are their own CLI flags, so their code stays
   unmodified. Don't parallelise that step.

3. **Three environment pins or their code will not run at all:** `pandas<3`
   (`np.array_split` on a DataFrame changed behaviour), `nnsight==0.5.15` (their
   fidelity call is rejected by every version 0.4.11–0.7.0), and `SSL_CERT_FILE`.
   `setup` applies all three.

4. **Read `sae_quality.txt` before `concept_f1.txt`.** It is written for every
   dictionary *before* any concept number exists for that cell, deliberately — a
   configuration must never be selected on the metric being reported. On our 33.2M pair
   the masked arm had **502 live features of 1,920** against causal's **1,551**, and the
   concept-F1 gap **flipped sign** with the L1 setting. If your 42M runs show comparable
   live-feature counts, the objective contrast is interpretable. If they don't, the
   non-comparability is itself the finding and we report that instead.

## Expected oddities

- **`compare_activations` is single-threaded** and will use one core for a long time.
  That is their code, not a misconfiguration. Cells are independent so `xargs -P N` over
  them is safe — but size N against RAM, not cores: each peaks at 12–15 GB, and 6-way on
  a 128 GB machine turned a 5-minute job into 187 minutes with nothing written. N = 3–4.
- **Zero concepts above F1 0.5 is possible** and does not mean a broken run. Our small
  local SAEs scored 0.067–0.137 with none identified, against 0.209 / 9 concepts for
  their own ESM-2 walkthrough. It means the regime has no discriminative power, which is
  itself reportable — but check `sae_quality.txt` before concluding anything.

## Two things I owe you

- **The `PY` quoting bug is ours, you were right, and it is now actually fixed** — we had
  left it in and relied on your environment override. `PY="${PY:-$PY_BIN -u}"` makes a
  single two-word string, so the 20 correctly-quoted call sites try to exec a file
  literally named `python -u`; the 4 unquoted ones word-split by luck, which is why it hid
  in some stages and not others. `RUN_MUSTRUNS.sh` now keeps `PY` a single word and gets
  unbuffering from `PYTHONUNBUFFERED=1`. You no longer need the override, and
  `./preflight.sh env` fails if the pattern ever comes back. This new runner never had it.
- **`--arch esmc` is smoke-tested, not grid-tested.** It ran against a 42M-architecture
  checkpoint with `esm==3.2.3` and passed (`SMOKE OK: (173, 320) rows, 8 proteins,
  sum(len)=173`), and `hiddens` was confirmed from esm's source to have exactly
  `n_layers` entries, so `hid[layer]` matches `eval_ctrl_plm.py` with no off-by-one.
  `smoke` on your own checkpoint is the two-minute check before committing the grid.

`interplm_attack/README.md` has the full detail, including the four checks to eyeball
and the six comparisons this feeds.

## If you have already started — one number I need

My ~68 h figure was measured on an **Apple M-series laptop**. `compare_activations` is
single-threaded CPU, so your RTX PRO 6000 does nothing for it and the only thing that
matters is single-core CPU speed. Server CPUs and Apple silicon differ enough on this kind
of work that the true figure on your box could plausibly be anywhere from half to double
mine. I have no honest way to predict it from here.

You can settle it in about a minute, and it is worth doing at hour 1 rather than hour 60:

```bash
# after the first compare_activations has been going a few minutes
grep -aoE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" $BASE/results/score_*.log | tail -3
```

That prints the tqdm pacing, e.g. `12/50 [14:06<44:38`. Then:

```
seconds_per_chunk x n_chunks x 8 shards x 81 cells / 3600 = hours
```

On mine: 72 s/chunk x 50 chunks x 8 x 81 / 3600 = 65 h.

**If yours comes out above ~4 days, stop and tell me before continuing.** The honest levers
are then SAE seeds 3 -> 1 (threefold), layers 3 -> 2, or dropping the untrained arm — all
of them decisions to take before burning the time, not after.

**Note:** if you pulled before 2026-08-07 evening, that step wrote to `/dev/null` and the
log will not exist. `git pull` to get the version that keeps it, or just time a full cell
end to end and divide by 8 to get per-shard.
