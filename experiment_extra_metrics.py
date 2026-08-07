#!/usr/bin/env python3
"""
experiment_extra_metrics.py — three published metrics that are NOT InterPLM's,
run on our layer dirs against the SAME concept set.

WHY THIS EXISTS
---------------
Everything we have shown so far attacks two measures: our own `L_struct` and
InterPLM's structural clustering / concept-F1. A reviewer's first question is
whether the failure is specific to those two. This runs three further metrics,
each from a different paper, on identical inputs.

The concept set, the fold-disjoint val/test split and the layer dir are shared
across all three, so the *only* thing that varies between them is the metric.
That is what makes a difference between them attributable to the metric rather
than to the data.

THE THREE METRICS
-----------------
1. `adams`     Adams et al., InterProt (ICML 2025). Single best latent per
               concept, plain per-residue F1, **no domain-level adjustment**.
               InterPLM's `calculate_f1.py` divides recall by domain
               (`recall_per_domain`); Adams' does not. Running both on the same
               concepts isolates the domain adjustment as a variable -- if the
               no-model baseline beats the SAE here but not under InterPLM, the
               domain adjustment is what protects them, which is a mechanism and
               not just another failed metric.

2. `geometry`  Li et al., "The Geometry of Concepts" (Entropy 2025, 27, 344).
               Do co-activating features sit close together in decoder space?
               Reported against **their own random-direction null**. We test
               whether that null actually discriminates.

3. `ksparse`   k-sparse probing (SAEBench, Karvonen et al. ICML 2025). Can the
               top-k SAE features linearly predict a concept? Swept over k. This
               is the SAE-side analogue of the pairwise probe result, where the
               raw residual stream already beat the SAE features.

WHAT MAKES THIS AN ATTACK
-------------------------
Point any of them at `make_synthetic_layer.py`'s output -- a layer dir whose
"features" are residue-identity indicators, containing no model at all. A metric
that cannot separate that from a trained dictionary is not measuring learned
structure. Point them at the shuffled-corpus arm for the relation-destroying
control. Both are the same discipline used on InterPLM, applied wider.

GUARDS (each one exists because something already failed silently)
------------------------------------------------------------------
- `load_layer` refuses if sum(lengths.npy) != Z rows.
- Dictionary quality is printed BEFORE any metric number, always. A
  configuration must never be selected on the metric being reported.
- The val/test split is FOLD-DISJOINT by default, reusing
  `experiment_concept_f1.split_proteins`, so no close homolog spans both.
- Every F1 is reported next to the prevalence floor (predict-everything), because
  a rare concept makes any F1 look good.
- Occupancy is checked against the top-k gate. A binary feature whose occupancy
  exceeds the gate has percentile exactly 1.0, nothing is strictly greater, and
  the score is silently 0 (this is the `--no-trivial` trap in
  make_synthetic_layer.py).
- `--jitter` breaks argmax ties, because binary features tie everywhere and
  np.argmax then pins every anchor to the N-terminus. That artefact produced a
  spurious published-looking result once already.

USAGE
-----
  python experiment_extra_metrics.py --layer-dir outputs_ctrl/ckpt_mlm_s42_token/layer_14 \
      --metric all --out results_extra/mlm_s42_L14

  # the attack: same command, no-model input
  python experiment_extra_metrics.py --layer-dir outputs_synthetic/composition \
      --metric all --jitter 1e-3 --out results_extra/synthetic
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cpu_stage import load_layer, load_phys_features
from experiment_concept_f1 import (
    build_offsets,
    build_concept_membership,
    parse_scope_concepts,
    split_proteins,
)

QUANTILES = (0.90, 0.95, 0.975, 0.99, 0.995, 0.999)

# Peak memory is rows x features x 4 bytes, and several arrays of that shape are
# live at once. A fixed row chunk therefore scales peak RSS linearly with the
# dictionary size: 50k rows x 10,240 features x 4 B = 2 GB per array, measured at
# 12.5 GB peak before this was made adaptive. Target ~200 MB per array instead.
_TARGET_BYTES = 200_000_000


def _row_chunk(n_feat):
    return max(1_000, min(50_000, _TARGET_BYTES // (4 * max(1, n_feat))))


# --------------------------------------------------------------------------
# shared setup
# --------------------------------------------------------------------------
def dictionary_quality(Z, gate_frac):
    """Outcome-blind. Printed before any metric number, never after."""
    n_res, n_feat = Z.shape
    ROW_CHUNK = _row_chunk(n_feat)
    live = np.zeros(n_feat, dtype=bool)
    nnz = 0
    occ = np.zeros(n_feat, dtype=np.int64)
    for lo in range(0, n_res, ROW_CHUNK):
        blk = np.asarray(Z[lo:lo + ROW_CHUNK], dtype=np.float32)
        act = blk > 0
        live |= act.any(0)
        occ += act.sum(0)
        nnz += int(act.sum())
    occupancy = occ / n_res
    return {
        "n_residues": int(n_res),
        "n_features": int(n_feat),
        "live_features": int(live.sum()),
        "dead_features": int(n_feat - live.sum()),
        "L0_mean": nnz / n_res,
        "max_occupancy": float(occupancy.max()),
        "n_over_gate": int((occupancy > gate_frac).sum()),
        "_occupancy": occupancy,
    }


def load_concepts(layer_dir, fasta_path, features_csv, levels, min_domains):
    Z, uids, lengths = load_layer(Path(layer_dir))
    uids = [str(u) for u in uids]
    # build_offsets returns (dict uid -> row offset, total rows)
    offsets, total = build_offsets(uids, lengths)
    if total != int(Z.shape[0]):
        raise ValueError(f"offset total {total} != Z rows {Z.shape[0]}")
    df_phys = load_phys_features(Path(features_csv))
    uid_concepts = parse_scope_concepts(Path(fasta_path), levels)
    concepts = build_concept_membership(
        uids, lengths, offsets, df_phys, uid_concepts,
        levels, include_ss=True, include_rsa=True,
    )
    concepts = {c: v for c, v in concepts.items()
                if len(np.unique(v[1])) >= min_domains}
    val_uids, test_uids = split_proteins(
        uids, seed=42, val_frac=0.5, fasta_path=fasta_path, level="fold")
    n_res = int(Z.shape[0])
    val_mask = np.zeros(n_res, dtype=bool)
    test_mask = np.zeros(n_res, dtype=bool)
    for u, L in zip(uids, lengths):
        off = offsets[u]
        (val_mask if u in val_uids else test_mask)[off:off + int(L)] = True
    if not (val_mask.any() and test_mask.any()):
        raise ValueError("one split is empty — check the fold assignment")
    return Z, uids, lengths, offsets, concepts, val_mask, test_mask


def _prevalence_floor(concept_mask, split_mask):
    """F1 of the predict-everything classifier. Any F1 must be read against it."""
    p = float((concept_mask & split_mask).sum()) / max(1, int(split_mask.sum()))
    return (2 * p / (p + 1)) if p > 0 else 0.0


# --------------------------------------------------------------------------
# metric 1 — Adams et al., single latent, per-residue F1, no domain adjustment
# --------------------------------------------------------------------------
def _thresholds(Z, quantiles, feat_chunk=None):
    """Candidate thresholds per feature.

    Row 0 is ALWAYS 0.0 -- "active at all". Without it a binary feature is
    unscoreable: its 0.90 quantile is 1.0, `> 1.0` never fires, and every F1 is
    silently 0. That is the same top-k-gate failure documented in
    make_synthetic_layer.py, and this metric walked straight into it on the
    first run.

    The remaining rows are quantiles over the POSITIVE activations only, so a
    sparse feature's thresholds are spread over the values it actually takes
    rather than over a mass of zeros.

    Chunked over features, not rows, so peak memory is bounded by feat_chunk.
    """
    n_res, n_feat = Z.shape
    if feat_chunk is None:
        feat_chunk = max(16, min(512, _TARGET_BYTES // (4 * max(1, n_res))))
    T = np.zeros((len(quantiles) + 1, n_feat), dtype=np.float32)
    for f0 in range(0, n_feat, feat_chunk):
        f1 = min(f0 + feat_chunk, n_feat)
        col = np.asarray(Z[:, f0:f1], dtype=np.float32)
        pos = np.where(col > 0, col, np.nan)
        with np.errstate(invalid="ignore"):
            q = np.nanquantile(pos, list(quantiles), axis=0)
        T[1:, f0:f1] = np.nan_to_num(q, nan=0.0)
        del col, pos
    return T


def metric_adams(Z, concepts, val_mask, test_mask, jitter, rng):
    n_res, n_feat = Z.shape
    ROW_CHUNK = _row_chunk(n_feat)
    thresholds = _thresholds(Z, QUANTILES)
    qlabels = ("active",) + tuple(str(q) for q in QUANTILES)

    cnames = list(concepts.keys())
    cmasks = np.zeros((len(cnames), n_res), dtype=bool)
    for i, c in enumerate(cnames):
        cmasks[i, concepts[c][0]] = True

    best = {c: {"val_f1": -1.0, "feature": -1, "q": None} for c in cnames}
    for qi in range(thresholds.shape[0]):
        pred_pos = np.zeros(n_feat, dtype=np.int64)
        tp = np.zeros((len(cnames), n_feat), dtype=np.int64)
        for lo in range(0, n_res, ROW_CHUNK):
            hi = min(lo + ROW_CHUNK, n_res)
            blk = np.asarray(Z[lo:hi], dtype=np.float32)
            if jitter:
                blk = blk + rng.normal(0, jitter, blk.shape).astype(np.float32) * (blk > 0)
            sel = val_mask[lo:hi]
            if not sel.any():
                continue
            pblk = (blk[sel] > thresholds[qi][None, :]).astype(np.float32)
            del blk
            pred_pos += pblk.sum(0).astype(np.int64)
            cm_f = cmasks[:, lo:hi][:, sel].astype(np.float32)
            tp += (cm_f @ pblk).astype(np.int64)
            del pblk, cm_f
        for i, c in enumerate(cnames):
            n_true = int((cmasks[i] & val_mask).sum())
            denom = pred_pos + n_true
            f1 = np.where(denom > 0, 2.0 * tp[i] / np.maximum(denom, 1), 0.0)
            j = int(np.argmax(f1))
            if f1[j] > best[c]["val_f1"]:
                best[c] = {"val_f1": float(f1[j]), "feature": j, "q": qlabels[qi],
                           "thresh": float(thresholds[qi][j])}

    rows = []
    for i, c in enumerate(cnames):
        b = best[c]
        if b["feature"] < 0:
            continue
        tp = fp = fn = 0
        for lo in range(0, n_res, ROW_CHUNK):
            hi = min(lo + ROW_CHUNK, n_res)
            sel = test_mask[lo:hi]
            if not sel.any():
                continue
            a = np.asarray(Z[lo:hi, b["feature"]], dtype=np.float32)[sel]
            pos = a > b["thresh"]
            truth = cmasks[i, lo:hi][sel]
            tp += int((pos & truth).sum()); fp += int((pos & ~truth).sum())
            fn += int((~pos & truth).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        rows.append({
            "concept": c, "feature": b["feature"], "quantile": b["q"],
            "val_f1": b["val_f1"], "test_f1": f1,
            "prevalence_floor": _prevalence_floor(cmasks[i], test_mask),
            "n_domains": int(len(np.unique(concepts[c][1]))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# metric 2 — Li et al., co-activation vs decoder geometry, random-direction null
# --------------------------------------------------------------------------
def metric_geometry(Z, sae_path, rng, n_null=20, max_pairs=400_000):
    from scipy.stats import spearmanr

    n_res, n_feat = Z.shape
    ROW_CHUNK = _row_chunk(n_feat)
    # Accumulate B^T B in row chunks. Materialising B costs n_res x n_feat x 4
    # bytes -- 12 GB at 10k features -- which is how the box died before.
    co = np.zeros((n_feat, n_feat), dtype=np.float32)
    tmp = np.empty((n_feat, n_feat), dtype=np.float32)   # reused, not reallocated
    counts = np.zeros(n_feat, dtype=np.float32)
    for lo in range(0, n_res, ROW_CHUNK):
        hi = min(lo + ROW_CHUNK, n_res)
        b = (np.asarray(Z[lo:hi], dtype=np.float32) > 0).astype(np.float32)
        np.matmul(b.T, b, out=tmp)
        co += tmp
        counts += b.sum(0)
        del b
    del tmp
    # Sample feature PAIRS directly. np.triu_indices(n_feat) materialises
    # n_feat^2/2 index pairs -- 52M pairs = 840 MB of int64 at 10k features, the
    # single largest allocation here -- and the full Jaccard matrix would be two
    # more arrays of n_feat^2. Both are avoided by indexing only what is sampled.
    n_all = n_feat * (n_feat - 1) // 2
    if n_all <= max_pairs:
        iu = np.triu_indices(n_feat, k=1)
    else:
        over = int(max_pairs * 2.2)
        i = rng.integers(0, n_feat, size=over)
        j = rng.integers(0, n_feat, size=over)
        keep = i < j
        iu = (i[keep][:max_pairs], j[keep][:max_pairs])
    co_v = co[iu]
    den_v = counts[iu[0]] + counts[iu[1]] - co_v
    jac_v = np.where(den_v > 0, co_v / np.maximum(den_v, 1e-9), 0.0).astype(np.float32)
    del co, co_v, den_v

    out = {"n_features": int(n_feat), "n_pairs": int(len(jac_v)),
           "mean_jaccard": float(jac_v.mean()),
           "frac_pairs_cooccur": float((jac_v > 0).mean())}

    D = _load_decoder(sae_path, n_feat) if sae_path else None
    if D is None:
        out["decoder"] = "absent — co-activation structure only"
        return out, pd.DataFrame({"jaccard": jac_v})

    Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    cos_v = np.einsum("ij,ij->i", Dn[iu[0]], Dn[iu[1]])
    rho = spearmanr(jac_v, cos_v).statistic
    out["rho_cooccur_vs_decoder_cos"] = float(rho)

    null = []
    for _ in range(n_null):
        R = rng.normal(size=D.shape).astype(np.float32)
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
        null.append(float(spearmanr(jac_v, np.einsum("ij,ij->i", R[iu[0]], R[iu[1]])).statistic))
    out["null_mean"] = float(np.mean(null))
    out["null_sd"] = float(np.std(null))
    out["z_vs_null"] = float((rho - np.mean(null)) / max(np.std(null), 1e-9))
    return out, pd.DataFrame({"jaccard": jac_v, "decoder_cos": cos_v})


def _load_decoder(sae_path, n_feat):
    import torch
    obj = torch.load(sae_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    for k in ("decoder.weight", "W_dec", "decoder", "dec.weight"):
        if isinstance(sd, dict) and k in sd:
            W = np.asarray(sd[k], dtype=np.float32)
            if W.shape[0] == n_feat:
                return W
            if W.shape[1] == n_feat:
                return W.T
    print(f"  ! decoder not found in {sae_path}; keys={list(sd)[:8]}", file=sys.stderr)
    return None


# --------------------------------------------------------------------------
# metric 3 — k-sparse probing (SAEBench)
# --------------------------------------------------------------------------
def metric_ksparse(Z, concepts, val_mask, test_mask, ks=(1, 2, 4, 8, 16, 32)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    n_res, n_feat = Z.shape
    ROW_CHUNK = _row_chunk(n_feat)
    rows = []
    for c, (res_idx, _dom) in concepts.items():
        cm = np.zeros(n_res, dtype=bool); cm[res_idx] = True
        yv, yt = cm[val_mask], cm[test_mask]
        if yv.sum() < 50 or yt.sum() < 50 or yv.all() or yt.all():
            continue
        # Rank features by mean difference ON VAL ONLY, in row chunks -- never
        # materialise the full val slice, which is n_val x n_feat x 4 bytes.
        s_pos = np.zeros(n_feat, dtype=np.float64)
        s_neg = np.zeros(n_feat, dtype=np.float64)
        n_pos = n_neg = 0
        for lo in range(0, n_res, ROW_CHUNK):
            hi = min(lo + ROW_CHUNK, n_res)
            sel = val_mask[lo:hi]
            if not sel.any():
                continue
            blk = np.asarray(Z[lo:hi], dtype=np.float32)[sel]
            y = cm[lo:hi][sel]
            s_pos += blk[y].sum(0); n_pos += int(y.sum())
            s_neg += blk[~y].sum(0); n_neg += int((~y).sum())
            del blk
        if n_pos == 0 or n_neg == 0:
            continue
        order = np.argsort(-np.abs(s_pos / n_pos - s_neg / n_neg))
        kmax = min(max(ks), n_feat)
        idx_all = order[:kmax]
        Xv = np.asarray(Z[:, idx_all], dtype=np.float32)[val_mask]
        Xt = np.asarray(Z[:, idx_all], dtype=np.float32)[test_mask]
        for k in ks:
            if k > n_feat:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xv[:, :k], yv)
            rows.append({
                "concept": c, "k": k,
                "test_auroc": float(roc_auc_score(yt, clf.predict_proba(Xt[:, :k])[:, 1])),
                "prevalence": float(yt.mean()),
            })
        del Xv, Xt
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="all",
                    choices=["all", "adams", "geometry", "ksparse"])
    ap.add_argument("--sae", default=None, help="ae.pt, for the geometry decoder")
    ap.add_argument("--fasta-path", default="cache/scope_40.fa")
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--levels", default="class,fold,superfamily,family")
    ap.add_argument("--min-domains", type=int, default=10)
    ap.add_argument("--gate-frac", type=float, default=0.02,
                    help="top-k gate to check occupancy against")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="tie-break for binary features; use 1e-3 on synthetic input")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    levels = [s for s in args.levels.split(",") if s]

    t0 = time.time()
    Z, uids, lengths, offsets, concepts, val_mask, test_mask = load_concepts(
        args.layer_dir, args.fasta_path, args.features_csv, levels, args.min_domains)

    # ---- outcome-blind, printed FIRST, always
    q = dictionary_quality(Z, args.gate_frac)
    occ = q.pop("_occupancy")
    print("== dictionary quality (outcome-blind, before any metric) ==")
    for k, v in q.items():
        print(f"   {k:18} {v}")
    if q["n_over_gate"]:
        print(f"   !! {q['n_over_gate']} feature(s) exceed gate_frac={args.gate_frac}. "
              f"A top-k gated metric scores these as exactly 0 (silently).")
    if q["live_features"] == 0:
        sys.exit("ABORT: no live features — the layer dir is empty or mis-read.")
    (out / "dictionary_quality.json").write_text(json.dumps(q, indent=2))
    np.save(out / "occupancy.npy", occ)
    print(f"   concepts kept: {len(concepts)}  "
          f"val residues {int(val_mask.sum()):,} / test {int(test_mask.sum()):,}")

    if args.metric in ("all", "adams"):
        t = time.time()
        df = metric_adams(Z, concepts, val_mask, test_mask, args.jitter, rng)
        df.to_csv(out / "adams_single_latent_f1.csv", index=False)
        beat = int((df.test_f1 <= df.prevalence_floor).sum())
        print(f"\n== adams (single latent, no domain adjustment)  [{time.time()-t:.0f}s] ==")
        print(f"   mean test F1 {df.test_f1.mean():.4f} over {len(df)} concepts; "
              f"{beat}/{len(df)} at or below the prevalence floor")

    if args.metric in ("all", "geometry"):
        t = time.time()
        summ, pairs = metric_geometry(Z, args.sae, rng)
        (out / "geometry_summary.json").write_text(json.dumps(summ, indent=2))
        pairs.sample(min(len(pairs), 50_000), random_state=args.seed).to_csv(
            out / "geometry_pairs_sample.csv", index=False)
        print(f"\n== geometry (Li et al.)  [{time.time()-t:.0f}s] ==")
        for k, v in summ.items():
            print(f"   {k:28} {v}")

    if args.metric in ("all", "ksparse"):
        t = time.time()
        df = metric_ksparse(Z, concepts, val_mask, test_mask)
        df.to_csv(out / "ksparse_probe.csv", index=False)
        print(f"\n== k-sparse probing (SAEBench)  [{time.time()-t:.0f}s] ==")
        if len(df):
            print(df.groupby("k").test_auroc.mean().to_string())

    print(f"\nwrote {out}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
