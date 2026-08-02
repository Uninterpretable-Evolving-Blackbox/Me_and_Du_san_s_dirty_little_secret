#!/usr/bin/env python3
"""
experiment_contact_null.py — matched contact-map nulls.

WHY THIS EXISTS
---------------
L_struct compares co-activation over a real contact graph against a null built by
permuting ACTIVATIONS while holding the graph fixed. That null controls for the
activation distribution. It does not control for the GRAPH: any property with
spatial autocorrelation in the fold -- burial, hydrophobicity, contact degree --
produces excess co-activation over a real contact map without the model encoding
anything structural.

So this permutes the other side. Keep the activations exactly as they are, and
replace the contact graph with one matched on its non-structural statistics. If
L_struct survives against a matched graph null, it is not reading real spatial
structure; it is reading whatever the matched statistic preserved.

THE NULLS
  separation -- preserves, exactly: the protein each edge belongs to, and the
                distribution of |i-j| over edges. Randomises WHICH residues are
                involved. This is the important one: near-diagonal contacts are
                57% of a 6 A contact set by edge count, so a graph matched on
                separation but otherwise random tests whether anything beyond the
                sequence-distance profile is doing work.
  degree     -- double-edge swap, preserving each residue's contact degree exactly.
                Separation is not preserved. Tests whether high-degree (buried)
                residues alone account for the effect.
There is deliberately no null preserving BOTH. A double-edge swap (a,b),(c,d) ->
(a,d),(c,b) produces separations d-a and b-c, which are not the originals even when
the two source edges had equal separation, so restricting swaps by separation does
not make them separation-preserving. That was written, tested, found not to do what
it claimed, and removed rather than shipped as a mislabelled control. The two nulls
above bracket the question between them.

USAGE
  python experiment_contact_null.py --layer-dir outputs_ctrl/ckpt_clm_s42/layer_14 \
      --nulls separation,degree --out results_contact_null/clm_s42_L14.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cpu_stage import (
    load_ref_seqs, build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
    _process_struct_seq_chunk_v3,
)


def edges_of(adj):
    return np.array([(r, n) for r, nbrs in enumerate(adj) for n in nbrs if n > r],
                    dtype=np.int64)


def protein_index(lengths):
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    p = np.zeros(int(np.sum(lengths)), dtype=np.int64)
    for k, (o, L) in enumerate(zip(offs, lengths)):
        p[o:o + int(L)] = k
    return p, offs


def null_separation(E, lengths, offs, prot, rng):
    """Preserve protein and |i-j| exactly; randomise position within the protein."""
    out = []
    for a, b in E:
        s = int(b - a); k = int(prot[a])
        lo, hi = int(offs[k]), int(offs[k]) + int(lengths[k])
        if hi - lo <= s:
            out.append((a, b)); continue
        i = int(rng.integers(lo, hi - s))
        out.append((i, i + s))
    return np.array(out, dtype=np.int64)


def null_swap(E, prot, rng, seq_gap=12, n_rounds=10):
    """Double-edge swap within each protein. Preserves each residue's degree exactly.
    Does NOT preserve the separation DISTRIBUTION -- see the module docstring -- but
    every rewired edge still satisfies |i-j| >= seq_gap, so the null never contains
    contacts the real graph was not allowed to contain.

    Two rejection rules, neither of which this originally had, both measured to be
    necessary. Without multi-edge rejection the swap created 9,522 duplicate edges,
    so degree was preserved only on the multigraph -- deduplicating changed 17,243
    residues' degrees. Without the floor check, 5% of rewired edges landed below
    |i-j| >= 12 with separations as low as 1, which is disqualifying here because
    the separation floor is the very thing this paper is about.
    """
    match_sep = False
    E = E.copy()
    existing = set(map(tuple, E))
    seps = E[:, 1] - E[:, 0]
    # Group by PROTEIN first. Pairing edges globally means two randomly chosen edges
    # are almost never in the same protein (~1/n_proteins), so essentially every swap
    # is rejected and the "null" comes back identical to the real graph. That failed
    # silently the first time this was run -- the swap rate is reported so it cannot
    # fail silently again.
    ep = prot[E[:, 0]]
    groups = {}
    for k in np.unique(ep):
        idx = np.where(ep == k)[0]
        if match_sep:
            for s in np.unique(seps[idx]):
                sub = idx[seps[idx] == s]
                if len(sub) >= 2:
                    groups[(k, s)] = sub
        elif len(idx) >= 2:
            groups[k] = idx
    tried = ok = 0
    for _ in range(n_rounds):
        for idx in groups.values():
            if len(idx) < 2:
                continue
            perm = rng.permutation(idx)
            for u, v in zip(perm[::2], perm[1::2]):
                a, b = E[u]; c, d = E[v]
                tried += 1
                if prot[a] != prot[c]:            # keep edges inside their protein
                    continue
                if len({a, b, c, d}) < 4:
                    continue
                na, nb = (a, d) if a < d else (d, a)
                nc, nd = (c, b) if c < b else (b, c)
                if na == nb or nc == nd:
                    continue
                if nb - na < seq_gap or nd - nc < seq_gap:
                    continue                          # never fall below the floor
                if (na, nb) in existing or (nc, nd) in existing:
                    continue                          # no multi-edges
                existing.discard(tuple(E[u])); existing.discard(tuple(E[v]))
                existing.add((na, nb)); existing.add((nc, nd))
                E[u] = (na, nb); E[v] = (nc, nd); ok += 1
    return E, (ok / max(tried, 1))


def adj_from_edges(E, n_res):
    adj = [[] for _ in range(n_res)]
    for a, b in E:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    return adj


def mean_struct_delta(Z, A_seq, deg_seq, A_struct, deg_struct, perms, n_jobs, topk):
    n_feat = int(Z.shape[1]); chunk = 256
    res = Parallel(n_jobs=n_jobs)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk, Z, None, A_seq, deg_seq, A_struct, deg_struct,
            perms, n_feat, topk)
        for ci in range((n_feat + chunk - 1) // chunk))
    obs = np.concatenate([r[2] for r in res]); sh = np.concatenate([r[4] for r in res])
    return float((obs - sh).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--seq-gap", type=int, default=12)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--nulls", default="separation,degree")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--topk-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.layer_dir)
    Z = np.asarray(np.load(d / "Z.npy"), dtype=np.float32)
    uids = [str(u) for u in json.loads((d / "uids.json").read_text())]
    lengths = np.load(d / "lengths.npy")
    ref_seqs = load_ref_seqs(d)
    n_res = int(np.sum(lengths))
    rng = np.random.default_rng(args.seed)

    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, lengths, ref_seqs, Path(args.pdb_dir), args.n_jobs,
        contact_cutoff=args.cutoff, seq_gap_min=args.seq_gap)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    perms = build_protein_permutations(lengths, args.n_shuffles)
    prot, offs = protein_index(lengths)
    E = edges_of(struct_adj)

    rows = []
    A_s, deg_s = adj_list_to_sparse(struct_adj, n_res)
    real = mean_struct_delta(Z, A_seq, deg_seq, A_s, deg_s, perms, args.n_jobs, args.topk_frac)
    print(f"  real graph      mean struct_delta {real:+.5f}   ({len(E):,} edges)")
    rows.append(dict(graph="real", mean_struct_delta=real, n_edges=len(E), swap_rate=None))

    for name in args.nulls.split(","):
        if name == "separation":
            En, rate = null_separation(E, lengths, offs, prot, rng), None
        elif name == "degree":
            En, rate = null_swap(E, prot, rng, seq_gap=args.seq_gap)
        else:
            continue
        A_n, deg_n = adj_list_to_sparse(adj_from_edges(En, n_res), n_res)
        v = mean_struct_delta(Z, A_seq, deg_seq, A_n, deg_n, perms, args.n_jobs, args.topk_frac)
        pct = 100 * v / real if real else float("nan")
        print(f"  null={name:11} mean struct_delta {v:+.5f}   ({pct:.0f}% of real)"
              + (f"   swap rate {rate:.2f}" if rate is not None else ""))
        rows.append(dict(graph=f"null_{name}", mean_struct_delta=v,
                         n_edges=len(En), swap_rate=rate))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"  written: {args.out}")
    print("  a null retaining most of the real value means L_struct is reading the")
    print("  matched statistic (separation profile or degree), not spatial structure.")


if __name__ == "__main__":
    main()
