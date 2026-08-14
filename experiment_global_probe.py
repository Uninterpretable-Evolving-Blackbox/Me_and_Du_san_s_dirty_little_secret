#!/usr/bin/env python3
"""experiment_global_probe.py — the GLOBAL level: remote homology between whole domains.

The three levels of the readout suite were meant to be position-wise, pairwise and global.
Position-wise is the linear probes (helix / strand / burial) and pairwise is
`experiment_pairwise_probe.py` (are two residues in contact). Global was never built. This
is it: given two whole domains, are they remote homologs?

DESIGN, and why this framing rather than fold classification
------------------------------------------------------------
Classifying fold directly is not usable here. 1,500 domains spread over ~430 SCOPe folds
leaves most folds with a handful of members, and a fold-disjoint split is impossible by
construction when the fold IS the label. So the global level uses the standard
remote-homology framing instead:

    positive pair : same FOLD, different SUPERFAMILY   (homologous, but not close)
    negative pair : different FOLD
    split         : SUPERFAMILY-disjoint -- no superfamily appears in both train and test

"Same fold, different superfamily" is what makes it *remote*: the pair cannot be solved by
recognising a close family resemblance, which is the failure mode that makes naive homology
probes easy.

Everything else deliberately mirrors `experiment_pairwise_probe.py`, so the three levels
differ only in the relation being predicted and not in the estimator:

  * same symmetric pair encoding, |a-b| concatenated with a*b
  * the same LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
    solver="liblinear", random_state=42)
  * the same shuffled-label control, drawn from its own generator so the label permutation
    cannot move the split (Hewitt & Liang require the split held fixed)
  * the same defaults, --seed 42 and --split-seed 1234
  * the same --mode sae|raw contrast, which is what makes "does the dictionary help?"
    answerable at this level too

Domain representation is the mean over the domain's residues. Mean pooling is the weakest
sensible choice, which is the point: a probe that succeeds on it is not relying on a clever
readout.

WHY THE SPLIT KEEPS BOTH ENDPOINTS, measured
--------------------------------------------
An earlier version of this file grouped each pair by the superfamily of its FIRST member,
copying the pairwise probe. That is safe there -- both residues of a contact are in the same
protein -- and wrong here, because the two domains of a pair are in different superfamilies by
construction, so a test pair's second domain could sit in a training superfamily.

Measured on ESM-2-35M layer 0: the leaky version reported **AUROC 0.880**, the strict version
**0.616**. The leak was worth **+0.264**.

The part worth remembering: the shuffled-label control did NOT catch it. It read 0.5045 leaky
and 0.5082 strict. A label permutation tests for label leakage; it is blind to GROUP leakage,
because permuting labels does not change which groups sit on which side of the split. Both
checks are needed and neither substitutes for the other.

Keeping only pairs whose endpoints fall on the same side costs test size -- about a third of
the pairs straddle the boundary and are dropped -- and `n_dropped_straddling` is reported so
that cost is visible rather than assumed.

Usage:
    python experiment_global_probe.py --layer-dir outputs_ctrl/ckpt_mlm_s42_token/layer_14 \
        --out results_global_probe/mlm_s42_L14
    python experiment_global_probe.py --layer-dir <dir> --mode raw --raw-npy <acts.npy> \
        --out results_global_probe/mlm_s42_L14_raw

CPU only. No GPU, no training of any model.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from experiment_concept_f1 import build_offsets
from cluster_bootstrap import load_uid_clusters


def pair_features(X, i, j):
    """Symmetric pair encoding, identical to the pairwise probe's."""
    a, b = X[i], X[j]
    return np.concatenate([np.abs(a - b), a * b], axis=1)


def pool_domains(X, uids, lengths, offsets):
    """One vector per domain: the mean over its residues.

    `offsets` is concept-F1's uid -> row-offset map, so the residue block for a domain is
    found by uid rather than by position; the two orders agree today but keying on the uid
    cannot silently misalign if they ever stop agreeing.
    """
    D = np.zeros((len(uids), X.shape[1]), dtype=np.float32)
    for i, u in enumerate(uids):
        s = int(offsets[str(u)]); L = int(lengths[i])
        if L > 0:
            D[i] = np.asarray(X[s:s + L], dtype=np.float32).mean(axis=0)
    return D


def sample_pairs(fold, sfam, rng, n_per_class):
    """positives: same fold, different superfamily. negatives: different fold."""
    n = len(fold)
    by_fold = {}
    for i, f in enumerate(fold):
        by_fold.setdefault(f, []).append(i)

    pos, tries = [], 0
    eligible = [f for f, m in by_fold.items() if len({sfam[i] for i in m}) >= 2]
    max_tries = 200 * max(1, n_per_class)
    if eligible:
        while len(pos) < n_per_class and tries < max_tries:
            tries += 1
            f = eligible[rng.integers(len(eligible))]
            m = by_fold[f]
            i, j = int(m[rng.integers(len(m))]), int(m[rng.integers(len(m))])
            if i != j and sfam[i] != sfam[j]:
                pos.append((i, j))
    if len(pos) < n_per_class:
        print(f"  only {len(pos)} remote-homolog pairs drawn in {tries} tries "
              f"(asked for {n_per_class}) — the eligible folds are the limit, not the sampler")

    neg, guard = [], 0
    while len(neg) < n_per_class and guard < 200 * max(1, n_per_class):
        guard += 1
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j and fold[i] != fold[j]:
            neg.append((i, j))
    k = min(len(pos), len(neg))
    return pos[:k], neg[:k]


def evaluate(D, pos, neg, sfam, rng, shuffle_labels=False, split_seed=1234):
    P, N = np.array(pos), np.array(neg)
    if len(P) == 0 or len(N) == 0:
        return dict(auroc=None, ap=None, n_train=0, n_test=0)
    F = np.vstack([pair_features(D, P[:, 0], P[:, 1]),
                   pair_features(D, N[:, 0], N[:, 1])])
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    # A pair has TWO endpoints and they are in different superfamilies by construction, so
    # grouping on the first member alone would leave a test pair's second domain sitting in a
    # training superfamily. Hold out superfamilies, then keep a pair only when BOTH of its
    # endpoints fall on the same side; pairs that straddle the boundary are dropped.
    a_sf = np.concatenate([sfam[P[:, 0]], sfam[N[:, 0]]])
    b_sf = np.concatenate([sfam[P[:, 1]], sfam[N[:, 1]]])

    split_rng = np.random.default_rng(split_seed)
    if shuffle_labels:
        y = rng.permutation(y)

    groups = np.unique(np.concatenate([a_sf, b_sf])); split_rng.shuffle(groups)
    te_g = set(groups[: max(1, len(groups) // 5)])
    in_te = np.array([(x in te_g) and (z in te_g) for x, z in zip(a_sf, b_sf)])
    in_tr = np.array([(x not in te_g) and (z not in te_g) for x, z in zip(a_sf, b_sf)])
    n_straddle = int(len(y) - in_te.sum() - in_tr.sum())
    te, tr = in_te, in_tr
    if te.sum() < 20 or tr.sum() < 20 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return dict(auroc=None, ap=None, n_train=int(tr.sum()), n_test=int(te.sum()),
                    n_dropped_straddling=n_straddle)

    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             solver="liblinear", random_state=42)
    clf.fit((F[tr] - mu) / sd, y[tr])
    s = clf.decision_function((F[te] - mu) / sd)
    return dict(auroc=float(roc_auc_score(y[te], s)),
                ap=float(average_precision_score(y[te], s)),
                n_train=int(tr.sum()), n_test=int(te.sum()),
                n_dropped_straddling=n_straddle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--mode", choices=["sae", "raw"], default="sae")
    ap.add_argument("--raw-npy", default=None, help="required when --mode raw")
    ap.add_argument("--fasta-path", default="cache/scope_40.fa")
    ap.add_argument("--n-pairs", type=int, default=5000,
                    help="per class. Lower than the pairwise probe's 20000 on purpose: there are\n                         only 1,500 domains here against ~294k residues, so 5,000 pairs per class\n                         already samples the space densely, while the pair matrix costs\n                         2*n_pairs x 2*n_features floats and 20000 makes it ~1 GB per cell.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    ld = Path(a.layer_dir)
    uids = [str(u) for u in json.loads((ld / "uids.json").read_text())]
    lengths = np.load(ld / "lengths.npy")
    offsets, _n_res = build_offsets(uids, lengths)

    if a.mode == "raw":
        if not a.raw_npy:
            raise SystemExit("--mode raw needs --raw-npy")
        X = np.load(a.raw_npy, mmap_mode="r")
    else:
        X = np.load(ld / "Z.npy", mmap_mode="r")

    fold_map = load_uid_clusters(a.fasta_path, level="fold")
    sfam_map = load_uid_clusters(a.fasta_path, level="superfamily")
    fold = np.array([fold_map.get(u, f"__f{i}") for i, u in enumerate(uids)])
    sfam = np.array([sfam_map.get(u, f"__s{i}") for i, u in enumerate(uids)])

    if X.shape[0] != _n_res:
        raise SystemExit(f"row mismatch: {a.mode} matrix has {X.shape[0]} rows but lengths.npy "
                         f"sums to {_n_res}. Pooling would silently misalign domains.")
    D = pool_domains(X, uids, lengths, offsets)
    print(f"  {a.mode}: {D.shape[0]} domains x {D.shape[1]} features "
          f"| {len(set(fold))} folds, {len(set(sfam))} superfamilies")

    rng = np.random.default_rng(a.seed)
    pos, neg = sample_pairs(fold, sfam, rng, a.n_pairs)
    print(f"  pairs: {len(pos)} remote-homolog, {len(neg)} different-fold")
    if not pos:
        print("  NO remote-homolog pairs available — every fold here has one superfamily.")
        print("  The global level is not measurable on this evaluation set.")

    real = evaluate(D, pos, neg, sfam, rng, shuffle_labels=False, split_seed=a.split_seed)
    ctrl = evaluate(D, pos, neg, sfam, rng, shuffle_labels=True, split_seed=a.split_seed)

    row = {"layer_dir": str(ld), "mode": a.mode,
           "n_pos": len(pos), "n_neg": len(neg),
           "auroc": real["auroc"], "ap": real["ap"],
           "auroc_shuffled_labels": ctrl["auroc"],
           "n_dropped_straddling": real.get("n_dropped_straddling"),
           "n_train": real["n_train"], "n_test": real["n_test"],
           "seed": a.seed, "split_seed": a.split_seed,
           "seconds": round(time.time() - t0, 1)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out / "global_probe.csv", index=False)

    if real["auroc"] is None:
        print("  not enough data after the superfamily-disjoint split — no AUROC.")
    else:
        print(f"  AUROC {real['auroc']:.4f}   AP {real['ap']:.4f}   "
              f"shuffled-label control {ctrl['auroc']:.4f}")
        print("  Read the control first: it should sit at ~0.5. Anything else means the split")
        print("  is leaking and the real number is not interpretable.")
    print(f"  wrote {out}/global_probe.csv  [{row['seconds']}s]")


if __name__ == "__main__":
    main()
