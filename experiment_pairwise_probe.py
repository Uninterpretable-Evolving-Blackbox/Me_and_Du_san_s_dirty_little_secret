#!/usr/bin/env python3
"""
experiment_pairwise_probe.py — is structure encoded at all? Ask it directly.

WHY THIS EXISTS
---------------
L_struct asks whether structurally related residues activate the SAME sparse
features. That is an assumption, not a definition, and it was never defended.
Structural information could equally sit in a subspace, be spread over many
features, or be encoded RELATIONALLY between two residue representations rather
than as shared activation of one. If it is relational, L_struct cannot see it, and
a null result says nothing about whether the model encodes structure.

This asks the question without that assumption: given the representations of two
residues, can a probe tell whether they are in contact? It is correlational, but
with the controls below it answers "is structure encoded at all", which
co-activation cannot.

Run on BOTH raw activations and SAE feature vectors. If raw pairs are predictable
and SAE pairs are not, the autoencoder is destroying the relation, which is a
different failure from the model not having it.

CONTROLS (all reported side by side; a number without them is uninterpretable)
  * shuffled-label     -- same pairs, permuted labels. The Hewitt & Liang control.
  * separation-matched -- negatives drawn to match the positives' |i-j| distribution,
                          so the probe cannot win by reading sequence distance alone.
                          This is the control that matters: without it a "contact
                          probe" is mostly a separation probe.
  * within-protein     -- pairs never cross proteins, and train/test split is BY
                          PROTEIN, so no protein appears in both.

USAGE
  python experiment_pairwise_probe.py --layer-dir outputs_ctrl/ckpt_clm_s42/layer_14 \
      --mode sae  --out results_pairwise_probe/clm_s42_L14_sae.json
  python experiment_pairwise_probe.py --raw-npy X.npy --layer-dir <same dir> \
      --mode raw --out results_pairwise_probe/clm_s42_L14_raw.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from cpu_stage import load_ref_seqs, build_neighbor_graphs_residue_parallel


def pair_features(X, i, j):
    """Symmetric pair encoding: |difference| and product.

    Symmetric because contact is an undirected relation -- an asymmetric encoding
    would let the probe learn residue ORDER, which is a different thing.
    """
    a, b = X[i], X[j]
    return np.concatenate([np.abs(a - b), a * b], axis=1)


def sample_pairs(struct_adj, lengths, offsets, rng, n_per_class, seq_gap,
                 match_separation=True):
    """Positive = contacting pair. Negative = non-contacting pair from the same
    protein, optionally matched on |i-j| so separation carries no signal."""
    pos = []
    for r, nbrs in enumerate(struct_adj):
        for n in nbrs:
            if n > r:
                pos.append((r, n))
    if not pos:
        raise SystemExit("no contacts in graph")
    # Build the rejection set from ALL contacts, BEFORE subsampling. Deriving it from
    # the truncated list lets any unsampled contact be drawn as a negative, which
    # biases AUROC and AP downward -- concentrated near the separation floor, exactly
    # where separation-matched sampling puts most of its mass.
    contact = set(pos) | {(b, a) for a, b in pos}
    rng.shuffle(pos)
    pos = pos[:n_per_class]

    prot_of = np.zeros(int(np.sum(lengths)), dtype=np.int64)
    for p, (o, L) in enumerate(zip(offsets, lengths)):
        prot_of[o:o + int(L)] = p
    neg = []
    for (a, b) in pos:
        sep_target = abs(b - a)
        p = prot_of[a]
        lo, hi = offsets[p], offsets[p] + int(lengths[p])
        for _ in range(40):
            if match_separation:
                i = rng.integers(lo, max(lo + 1, hi - sep_target))
                j = i + sep_target
            else:
                i, j = sorted(rng.integers(lo, hi, 2))
                if abs(j - i) < seq_gap:
                    continue
            if j < hi and i != j and (i, j) not in contact:
                neg.append((int(i), int(j))); break
    n = min(len(pos), len(neg))
    return pos[:n], neg[:n], prot_of


def evaluate(X, pos, neg, prot_of, rng, shuffle_labels=False, split_seed=1234):
    P = np.array(pos); N = np.array(neg)
    Fp = pair_features(X, P[:, 0], P[:, 1])
    Fn = pair_features(X, N[:, 0], N[:, 1])
    F = np.vstack([Fp, Fn])
    y = np.concatenate([np.ones(len(Fp)), np.zeros(len(Fn))])
    prot = np.concatenate([prot_of[P[:, 0]], prot_of[N[:, 0]]])
    # Draw the split from its OWN generator. Sharing one rng meant the label
    # permutation advanced it, so the control ran on a different protein partition
    # than the real fit -- the Hewitt & Liang control requires the split held fixed.
    split_rng = np.random.default_rng(split_seed)
    if shuffle_labels:
        y = rng.permutation(y)

    prots = np.unique(prot); split_rng.shuffle(prots)
    te_p = set(prots[: max(1, len(prots) // 5)])          # split BY PROTEIN
    te = np.array([p in te_p for p in prot])
    if te.sum() < 20 or (~te).sum() < 20 or len(np.unique(y[~te])) < 2:
        return dict(auroc=None, ap=None, n_train=int((~te).sum()), n_test=int(te.sum()))

    mu, sd = F[~te].mean(0), F[~te].std(0) + 1e-6
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             solver="liblinear", random_state=42)
    clf.fit((F[~te] - mu) / sd, y[~te])
    s = clf.decision_function((F[te] - mu) / sd)
    return dict(auroc=float(roc_auc_score(y[te], s)),
                ap=float(average_precision_score(y[te], s)),
                n_train=int((~te).sum()), n_test=int(te.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--mode", choices=["sae", "raw"], default="sae")
    ap.add_argument("--raw-npy", default=None, help="required when --mode raw")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--seq-gap", type=int, default=12)
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.layer_dir)
    rng = np.random.default_rng(args.seed)
    X = (np.load(args.raw_npy) if args.mode == "raw" else np.load(d / "Z.npy"))
    X = np.asarray(X, dtype=np.float32)

    uids = [str(u) for u in json.loads((d / "uids.json").read_text())]
    lengths = np.load(d / "lengths.npy")
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    ref_seqs = load_ref_seqs(d)

    _, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, lengths, ref_seqs, Path(args.pdb_dir), args.n_jobs,
        contact_cutoff=args.cutoff, seq_gap_min=args.seq_gap)

    res = {"layer_dir": str(d), "mode": args.mode, "dim": int(X.shape[1]),
           "cutoff": args.cutoff, "seq_gap": args.seq_gap}
    for tag, match in [("separation_matched", True), ("unmatched", False)]:
        pos, neg, prot_of = sample_pairs(struct_adj, lengths, offsets, rng,
                                         args.n_pairs, args.seq_gap, match)
        res[tag] = evaluate(X, pos, neg, prot_of, rng)
        res[tag + "_shuffled_labels"] = evaluate(X, pos, neg, prot_of, rng,
                                                 shuffle_labels=True)
        a = res[tag]["auroc"]; b = res[tag + "_shuffled_labels"]["auroc"]
        print(f"  {tag:20} AUROC {a if a is None else round(a,4)}   "
              f"shuffled-label control {b if b is None else round(b,4)}   "
              f"(n_pairs={len(pos)*2})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"  written: {args.out}")
    print("  read separation_matched, not unmatched. the shuffled-label control")
    print("  must sit at ~0.5; if it does not, the split is leaking.")


if __name__ == "__main__":
    main()
