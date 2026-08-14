#!/usr/bin/env python3
"""experiment_trivial_baseline.py — how much of a concept score is just label prevalence?

A binary concept score is only evidence of learned structure if it beats the classifier that
marks every residue positive. For a label with prevalence p that classifier scores
F1 = 2p/(p+1) -- 0.33 at p=0.20, 0.60 at p=0.42, 0.67 at p=0.50. Secondary-structure and
burial labels are prevalent enough that this floor is most of the reportable range, so a
score quoted without it says very little.

This measures, per cell and per label:

  * the label's prevalence on the held-out split, and the resulting floor
  * the best single feature's F1 (threshold chosen on val, reported on test)
  * the MARGIN of that F1 over the floor, which is the only part attributable to the model

Run it on the trained arms, on --random-init cells, and with --mode raw, and the three
together say whether a concept number reflects the representation or the label frequency.

Nothing here is reimplemented: the floor comes from `experiment_extra_metrics._prevalence_floor`,
the labels from `experiment_concept_f1`'s secondary-structure and RSA definitions, and the
split from `experiment_concept_f1.split_proteins` at fold level. If any of those change, this
changes with them.

Usage:
    python experiment_trivial_baseline.py --layer-dir outputs_ctrl/ckpt_mlm_s42_token/layer_14 \
        --out results_trivial_baseline/mlm_s42_L14
    python experiment_trivial_baseline.py --layer-dir <dir> --mode raw --raw-npy <acts.npy> \
        --out results_trivial_baseline/mlm_s42_L14_raw

CPU only. No GPU, no training.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_concept_f1 import (SS_GROUPS, split_proteins, build_offsets)
from experiment_extra_metrics import _prevalence_floor

LABELS = ("ss_helix", "ss_strand", "rsa_buried")


def load_labels(uids, lengths, offsets, n_res, features_csv, rsa_buried=0.1):
    """residue-level boolean mask per label, using concept-F1's own definitions."""
    df = pd.read_csv(features_csv)
    # concept-F1 reads the "sasa" column directly as RSA (experiment_concept_f1.py:193-196),
    # so the same column and the same 0.1 cutoff are used here rather than a second definition.
    need = {"uid", "ss_8class", "sasa"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{features_csv} lacks {missing}; regenerate it with cpu_stage.py")
    total = int(n_res)
    masks = {k: np.zeros(total, dtype=bool) for k in LABELS}
    by_uid = {u: g for u, g in df.groupby("uid")}
    for i, u in enumerate(uids):
        g = by_uid.get(str(u))
        if g is None:
            continue
        L = int(lengths[i]); base = int(offsets[str(u)])
        n = min(L, len(g))
        ss = g["ss_8class"].astype(str).to_numpy()[:n]
        rsa = pd.to_numeric(g["sasa"], errors="coerce").to_numpy()[:n]
        sl = slice(base, base + n)
        masks["ss_helix"][sl] = np.isin(ss, list(SS_GROUPS["ss_helix"]))
        masks["ss_strand"][sl] = np.isin(ss, list(SS_GROUPS["ss_strand"]))
        with np.errstate(invalid="ignore"):
            masks["rsa_buried"][sl] = rsa < rsa_buried

    # cpu_stage.py fills ss_8class from DSSP only when it is present; an unfilled column is
    # all "-", which would give prevalence 0.000 and a floor of 0.000 -- a number that looks
    # like a result. Fail loudly instead.
    for k, m in masks.items():
        if m.sum() == 0:
            raise SystemExit(
                f"label {k!r} has ZERO positive residues in {features_csv}. That is a broken "
                f"annotation, not a finding -- ss_8class is probably unfilled ('-' everywhere). "
                f"Regenerate it with cpu_stage.py before trusting anything here.")
    return masks


def best_single_feature_f1(X, label, val_mask, test_mask, n_quantiles=9, chunk=256):
    """Best (feature, threshold) by val F1; that exact pair re-scored on test.

    Selecting on val and reporting on test is what stops the number being the maximum of a
    large number of noisy draws -- with thousands of features, the best TEST F1 chosen on
    test is an order statistic, not a measurement.
    """
    qs = np.linspace(0.1, 0.9, n_quantiles)
    best = {"val_f1": -1.0, "test_f1": 0.0, "feature": -1, "quantile": None}
    yv, yt = label[val_mask], label[test_mask]
    if yv.sum() == 0 or yt.sum() == 0:
        return best
    for c0 in range(0, X.shape[1], chunk):
        blk = np.asarray(X[:, c0:c0 + chunk], dtype=np.float32)
        for j in range(blk.shape[1]):
            col = blk[:, j]
            nz = col[col > 0]
            if nz.size < 10:
                continue
            for q in qs:
                thr = float(np.quantile(nz, q))
                pv = col[val_mask] > thr
                tp = float(np.logical_and(pv, yv).sum())
                if tp == 0:
                    continue
                f1v = 2 * tp / (pv.sum() + yv.sum())
                if f1v > best["val_f1"]:
                    pt = col[test_mask] > thr
                    tpt = float(np.logical_and(pt, yt).sum())
                    f1t = 2 * tpt / (pt.sum() + yt.sum()) if (pt.sum() + yt.sum()) else 0.0
                    best = {"val_f1": f1v, "test_f1": f1t,
                            "feature": int(c0 + j), "quantile": float(q)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--mode", choices=["sae", "raw"], default="sae")
    ap.add_argument("--raw-npy", default=None, help="required when --mode raw")
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--fasta-path", default="cache/scope_40.fa")
    ap.add_argument("--split-level", default="fold", choices=["protein", "fold"])
    ap.add_argument("--seed", type=int, default=42, help="split seed; 42 matches concept-F1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ld = Path(a.layer_dir)
    uids = [str(u) for u in json.loads((ld / "uids.json").read_text())]
    lengths = np.load(ld / "lengths.npy")
    offsets, n_res = build_offsets(uids, lengths)

    if a.mode == "raw":
        if not a.raw_npy:
            raise SystemExit("--mode raw needs --raw-npy")
        X = np.load(a.raw_npy, mmap_mode="r")
    else:
        X = np.load(ld / "Z.npy", mmap_mode="r")
    if X.shape[0] != n_res:
        raise SystemExit(f"row mismatch: {a.mode} matrix has {X.shape[0]} rows but lengths.npy "
                         f"sums to {n_res}. The label masks would silently misalign.")
    print(f"  {a.mode}: {X.shape[0]} residues x {X.shape[1]} features")

    masks = load_labels(uids, lengths, offsets, n_res, a.features_csv)
    val_u, test_u = split_proteins(uids, seed=a.seed, val_frac=0.5,
                                   fasta_path=a.fasta_path, level=a.split_level)
    res_val = np.zeros(X.shape[0], dtype=bool)
    res_test = np.zeros(X.shape[0], dtype=bool)
    for i, u in enumerate(uids):
        sl = slice(int(offsets[str(u)]), int(offsets[str(u)]) + int(lengths[i]))
        if u in val_u:
            res_val[sl] = True
        elif u in test_u:
            res_test[sl] = True

    rows = []
    for lab in LABELS:
        t0 = time.time()
        m = masks[lab]
        floor = _prevalence_floor(m, res_test)
        prev = float((m & res_test).sum()) / max(1, int(res_test.sum()))
        b = best_single_feature_f1(X, m, res_val, res_test)
        rows.append({
            "label": lab, "mode": a.mode,
            "prevalence": prev, "prevalence_floor": floor,
            "best_feature": b["feature"], "quantile": b["quantile"],
            "val_f1": b["val_f1"], "test_f1": b["test_f1"],
            "margin_over_floor": b["test_f1"] - floor,
            "beats_floor": bool(b["test_f1"] > floor),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  {lab:<12} prevalence {prev:.3f}  floor {floor:.3f}  "
              f"best test F1 {b['test_f1']:.3f}  margin {b['test_f1']-floor:+.3f}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "trivial_baseline.csv", index=False)
    n_beat = int(df.beats_floor.sum())
    print(f"\n  {n_beat}/{len(df)} labels beat the prevalence floor; "
          f"margins {df.margin_over_floor.min():+.3f} to {df.margin_over_floor.max():+.3f}")
    print(f"  wrote {out}/trivial_baseline.csv")


if __name__ == "__main__":
    main()
