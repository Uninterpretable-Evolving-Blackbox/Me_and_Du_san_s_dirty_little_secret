# Running InterPLM's published metric on our controlled pair

**Everything here is InterPLM's own code. The only substitution is the backbone model.**

Three commands. Read §2 first — there are four checks you must eyeball, and each
one exists because the corresponding failure produces *plausible numbers* rather
than an error.

---

## 1. What to run

```bash
# --- prerequisites -----------------------------------------------------------
# SAE_SRC: the directory in YOUR checkout containing model_ctrl_esmc.py — the
#          same module train_ctrl_plm.py imports when you train the 42M models.
#          Nothing is copied from it; the adapter imports CtrlESMC from there.
export SAE_SRC=/path/to/your/checkout/src
export CKPT_ROOT=/home/ronnie/own_sae_data/uniref50_pilot          # REAL models
export CKPT_ROOT_SHUF=/home/ronnie/own_sae_data/<shuffled-tree>    # SHUFFLED models
export BASE=$HOME/interplm_stress                   # ~60 GB working dir

# Needs python3.12 on PATH (setup creates its own venv, isolated from yours) and
# esm==3.2.3 importable from SAE_SRC's environment for CtrlESMC. Put the four
# files in this directory anywhere; they reference each other relatively.

# --- three commands ----------------------------------------------------------
./RUN_INTERPLM_STRESS.sh setup     # ~2-3 h, mostly serial by design. Once.
./RUN_INTERPLM_STRESS.sh smoke     # ~2 min. Do not skip.
./RUN_INTERPLM_STRESS.sh grid      # the run
```

`CKPT_ROOT_SHUF` is a **different tree** from `CKPT_ROOT` —
`prep_controlled_corpus.py --shuffle-residues` writes its own out-dir. If you
don't have the shuffled checkpoints to hand, find them before starting; the
script will refuse to run the shuffled arms rather than silently substitute the
real ones (see check 1).

Defaults, override by exporting: `LAYERS="11 14 18"`, `SEEDS="0 1 2"`,
`L1=0.06`, `MAXLEN=510`, `DEV=cuda`.

---

## 2. The four checks

### Check 1 — after `grid` starts, confirm the preflight line

```
preflight OK: 8 checkpoints, all distinct
```

This sha256-hashes every checkpoint and aborts if two arms resolve to identical
weights. **Why it matters:** if the shuffled arms accidentally point at the real
checkpoints, the control compares the real models against themselves and reports
"shuffled ≈ real" — which looks like the metric *passing* a validity check. That
control is the paper's central claim. Path equality isn't enough, so this catches
copies and symlinks too.

If you see `distinct arms resolve to IDENTICAL checkpoint weights`, stop and fix
`CKPT_ROOT_SHUF`.

### Check 2 — after `setup`, confirm the generated trainer

```
SETUP COMPLETE (generated trainer verified)
```

`setup` builds `examples/train_ctrl_sae.py` from their `train_basic_sae.py` with
8 `sed` edits (paths, `d_model`, expansion, L1, seed, and `EvaluationConfig` in
place of the broken `ESMFidelityConfig`). A `sed` that matches nothing produces a
file that **runs fine with their hyperparameters instead of ours**. The script now
greps for each substitution and aborts if any is missing.

### Check 3 — before trusting any error bar, confirm the seeds actually varied

```bash
grep -h "seed" $BASE/../interplm_repo/models/grid/*_s{0,1,2}/config.yaml | sort -u
```

You should see **three distinct values**. If all three are `0`, the seeds never
reached the dictionary and you have n=1 copied three times, not n=3.

*(Background: InterPLM's `ActivationsDataLoader.__init__` calls
`torch.manual_seed(config.seed)`, and the dataloader is built at
`training_run.py:174` — before the SAE at line 211. An earlier version of this
package injected the seed at the top of `main()`, which was overwritten before a
single dictionary weight was drawn; all three "seeds" came out bit-identical.
Now routed through `DataloaderConfig(seed=…)`.)*

**Caveat for the write-up:** that seed controls dictionary initialisation **and**
shard/batch order jointly. There's no init-only knob without editing their
library, which the one rule forbids. So the honest phrasing is *"SAE replicates
vary dictionary initialisation and data order jointly."*

### Check 3b — paths, before you leave it running overnight

Five things worth confirming by eye in the first ten minutes, because each fails
late and silently rather than early and loudly:

```bash
ls $CKPT_ROOT $CKPT_ROOT_SHUF                    # both exist, and are DIFFERENT trees
ls $SAE_SRC/model_ctrl_esmc.py                   # the adapter imports CtrlESMC from here
ls $BASE/ann/processed/{valid,test}/metadata.json   # setup finished; if missing, setup died
df -h $BASE                                      # >= 60 GB free
nvidia-smi                                       # DEV=cuda is real
```

Then after the first cell starts, confirm output is actually appearing:

```bash
ls $BASE/../interplm_repo/results/grid/*_valid/  # non-empty within ~15 min
```

If that directory exists but stays empty for much longer, the run is memory-
thrashing rather than progressing — see §3 on parallelism. An empty directory with
a busy CPU is the signature.

### Check 4 — read `sae_quality.txt` BEFORE `concept_f1.txt`

```bash
column -t $BASE/results/sae_quality.txt      # var_expl  dead  live  L0  max_frac
```

`sae_quality.txt` is written for every dictionary *before* any concept number for
that cell exists. That ordering is deliberate: **never select a configuration on
the metric you're about to report.** Select on `var_expl`, `L0` and dead count only.

**What we found on the 33.2M pair, which is why this check exists:** at L1 = 0.06
the masked arm had **502 live features of 1,920** against the causal arm's
**1,551**, and the masked-vs-causal F1 gap **flipped sign** with the L1 setting
(causal > masked at 0.06, masked > causal at 0.02). Zero concepts cleared F1 > 0.5
in any cell.

So: if your 42M runs show *comparable* live-feature counts across arms, the
objective contrast is interpretable. If they don't, **the non-comparability is
itself the finding** and we report that instead of an objective effect. Either way
the F1 numbers mean nothing without the quality columns beside them.

---

## 3. Resource budget

| | |
|---|---|
| disk | **~60 GB** steady state. Embeddings are deleted per model, so peak is ~24 GB of embeddings + SAEs + results |
| memory, `setup` | bounded — `extract_annotations` runs `--max_workers 1` |
| memory, `grid` | ~12–15 GB per concurrent cell |
| GPU | 27 embedding passes (9 models × 3 layers), ~18M residues each |
| grid size | 9 models × 3 layers × 3 seeds = **81 cells** |

### Parallelism — measured, not guessed

The grid is **serial as shipped**. If you parallelise it, size N so that
`N × 15 GB` leaves real headroom, and **watch throughput, not memory pressure**.

We measured this the expensive way. `compare_activations` takes **~5 min** on the
valid split running alone. At **6-way concurrency on a 128 GB machine it took
187 minutes and had written nothing** — a >37× slowdown, with every worker still
at 92% CPU. Six processes × ~13 GB drove free memory to ~0 GB, macOS began
compressing, and each touch of the dense feature matrix then paid a decompress.
Swap and compressor-occupancy both stayed *flat* the whole time, so those are the
wrong things to watch — **watch whether output files are appearing.**

Rule of thumb on a 128 GB box: **N = 3–4**, and abort if a cell hasn't written
`results/grid/<tag>_valid/` contents within ~15 min.

**Never parallelise `extract_annotations`.** It defaults to
`ProcessPoolExecutor(os.cpu_count())` with each worker holding a fully exploded
per-residue DataFrame (15–30 GB each). That took the laptop down twice. `setup`
pins `--max_workers 1 --n_shards 16` — both their own CLI flags, so their code is
still unmodified.

---

## 4. What comes out

```
$BASE/results/
  sae_quality.txt          var_expl / dead / live / L0 / max_frac, per dictionary
  concept_f1.txt           avg best F1, concepts identified, features associated
  floor_vs_sae_<tag>.csv   per-concept residue-type floor vs best SAE feature
  train_<tag>.log
```

`floor_vs_sae_*.csv` is the one to look at second. For every concept InterPLM
reports, it gives the best F1 obtainable from **a single amino-acid indicator** —
best AA chosen on valid, scored on test, the same held-out protocol they use for
features. The number that matters is **floor ÷ SAE**, which is invariant to concept
prevalence. Do not compare raw floors across different protein sets: on 921
highly-annotated proteins Helix had a floor of 0.469, on 50k Swiss-Prot it's 0.048,
purely because prevalence drops from a large share to 1.84%.

---

## 5. The comparisons this feeds

| | contrast | varies | fixed |
|---|---|---|---|
| **A** | objective transfer | attention mask | corpus, batch order, init, SAE recipe |
| **B** | **validity control** | real vs shuffled corpus | everything else |
| **C** | untrained floor | trained vs random weights | architecture, tokeniser, SAE recipe |
| **D** | dictionary family | their ReLU/L1 vs our TopK k=256 e8 | model, layer |
| **E** | metric family | concept-F1 vs `L_struct` | model, layer |
| **F** | composition floor | — | needs no model at all |

**B is the important one.** Prespecified: if the metric names a relation and the
training data contains none of it, the metric must not rise. That's the number the
paper turns on, and it's the one a laptop has never been able to run.

---

## 6. Why this needs your box

- **3 model seeds per arm** (`mlm_s42/43/44`, `clm_s42/43/44`). The laptop has the
  older 33.2M pair at **n=1**, so there is currently no error bar on the
  masked-vs-causal contrast at all. The "seeds" it can vary are dictionary seeds.
- **The shuffled-corpus arm.** Never run locally — no shuffled 33.2M exists.
- **The 42.0M ESM-C models themselves** — the ones the paper is actually about.
  The laptop only has the earlier 33.2M implementation, and `L_struct` is known to
  flip sign between the two.

The `--arch esmc` adapter path has been run against a real 42M-architecture
checkpoint with `esm==3.2.3` and passes (`SMOKE OK: (173, 320) rows, 8 proteins,
sum(len)=173`), and `hiddens` was confirmed from esm's source to have exactly
`n_layers` entries, so `hid[layer]` has no off-by-one against
`sae_review/src/eval_ctrl_plm.py`. Still run `smoke` on your own checkpoint — it's
a confirmation now rather than a gamble.

---

## 7. Files, and what is and isn't ours

| file | role |
|---|---|
| `RUN_INTERPLM_STRESS.sh` | `setup` / `smoke` / `grid` |
| `embed_ctrl_interplm.py` | **the only substitution** — our models → their input format. Handles `CtrlESMC` (42M) and `PLM` (33.2M) |
| `sae_quality.py` | outcome-blind dictionary quality |
| `aa_floor.py` | composition floor and floor ÷ SAE ratio |

Nothing in `interplm_repo/` is edited. One file is *generated* —
`examples/train_ctrl_sae.py`, an 8-line diff from their own
`examples/train_basic_sae.py`. Every metric module runs unmodified as
`python -m interplm.analysis.concepts.*`.

### Known issues in their code we work around, not around us

- **`nnsight`**: their `ESMFidelityConfig` calls `NNsight(model, device=…)`, which
  every installable version (0.4.11 → 0.7.0) rejects. We use the base
  `EvaluationConfig`. This affects reconstruction *reporting* only, not concept-F1.
- **pandas ≥ 3** breaks `extract_annotations` (`np.array_split` on a DataFrame no
  longer returns DataFrames). Pinned to `<3`.
- **Their README's Option A** points at
  `data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz`, which is gitignored and
  absent from the repo. We use their documented Option B (UniProt REST).
- **Their SAE dataloader** silently rejects shards lacking a `metadata.yaml` that
  only their own extractor writes, and reports four candidate causes — none of
  them the real one. Our adapter writes it.
- **`interplm.sae.normalize`** at their current HEAD does not affect the F1
  numbers: `compare_activations` doesn't request normalised features, so the
  thresholds land on raw activations whose scale isn't comparable between
  backbones. It's their code, so we leave it — but report the distribution of
  `$SAVE/feature_stats/max.npy` per arm alongside every F1.
