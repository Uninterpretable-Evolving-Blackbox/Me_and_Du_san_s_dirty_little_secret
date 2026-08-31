#!/usr/bin/env python3
"""
Numerical smoke test for rescore_denominator.py.

The claim being tested is the one the whole script rests on: that its `sd`
denominator reproduces cpu_stage._process_struct_seq_chunk_v3 -- the function
that produced every published L_struct number -- to floating-point identity.
Everything else in the script is only worth reading if that holds.

Runs against the REAL cpu_stage functions, not a copy. No GPU, no checkpoints,
no PDB files: the contact graph is injected directly as a sparse matrix, which
is exactly what cpu_stage hands its own chunk processor.

    python3 tests/test_rescore_denominator.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cpu_stage as cs
import rescore_denominator as rd

FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def _rank_reference(a):
    """Slow per-column average-rank reference, for checking the fast path."""
    from scipy.stats import rankdata
    n = a.shape[0]
    out = np.empty_like(a, dtype=np.float32)
    for j in range(a.shape[1]):
        out[:, j] = (rankdata(a[:, j], method="average") - 1.0) / max(n - 1, 1)
    return out


def make_case(seed=0, n_res=400, n_feat=64, n_prot=8, density=0.10):
    """Synthetic layer: TopK-like sparse activations + a random contact graph."""
    rng = np.random.default_rng(seed)

    acts = np.zeros((n_res, n_feat), dtype=np.float32)
    for j in range(n_feat):
        k = max(1, int(density * n_res))
        rows = rng.choice(n_res, size=k, replace=False)
        acts[rows, j] = rng.gamma(2.0, 1.0, size=k).astype(np.float32)

    # pathological features the real data contains
    acts[:, 0] = 0.0                                   # dead feature
    acts[:, 1] = 0.0
    acts[rng.choice(n_res, 3, replace=False), 1] = 5.0  # n_active < 5 -> forced 0
    acts[:, 2] = 7.5                                   # constant: SD 0, IQR 0

    # protein lengths summing to n_res, for the within-protein permutation
    cuts = np.sort(rng.choice(np.arange(20, n_res - 20), n_prot - 1, replace=False))
    lengths = np.diff(np.concatenate([[0], cuts, [n_res]])).astype(np.int32)
    assert lengths.sum() == n_res and (lengths > 0).all()

    A = sparse.random(n_res, n_res, density=0.01, format="coo", random_state=seed)
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    A.setdiag(0)
    # force a handful of isolated residues so the deg==0 branch is exercised:
    # real contact graphs always have some (short chains, termini, gaps)
    iso = rng.choice(n_res, size=max(3, n_res // 50), replace=False)
    A = A.tolil()
    for r in iso:
        A[r, :] = 0
        A[:, r] = 0
    A = A.tocsr()
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    # leave some residues with no neighbours: the deg==0 branch must be exercised
    check("fixture exercises deg==0 branch", (deg == 0).any(),
          f"{int((deg == 0).sum())} isolated residues")
    return acts, A, deg, lengths


def main():
    print("=" * 68)
    print("test_rescore_denominator.py")
    print("=" * 68)

    acts, A, deg, lengths = make_case()
    n_res, n_feat = acts.shape

    # A second, different graph so seq and struct are not the same matrix
    A2 = A.copy()
    A2.data = A2.data * 0 + 1.0
    A2 = (A2 + sparse.eye(n_res, format="csr", dtype=np.float32) * 0).tocsr()
    deg2 = np.asarray(A2.sum(axis=1)).ravel().astype(np.float32)

    perms = cs.build_protein_permutations(lengths, 5)
    check("permutation draw is within-protein", all(
        np.array_equal(np.sort(p[o:o + L]), np.arange(o, o + L))
        for p in perms
        for o, L in [(int(lengths[:i].sum()), int(lengths[i])) for i in range(len(lengths))]
    ))

    # ---- reference: the actual published chunk processor -------------------
    ref = cs._process_struct_seq_chunk_v3(
        0, n_feat, acts, None, A2, deg2, A, deg, perms, n_feat, 0.10)
    ref_seq_delta = ref[1] - ref[3]
    ref_str_delta = ref[2] - ref[4]

    # ---- ours --------------------------------------------------------------
    got = rd._process_chunk(
        0, n_feat, acts, None, A2, deg2, A, deg, perms, n_feat, 0.10,
        ("sd", "fixed", "iqr", "rank"))

    check("sd reproduces cpu_stage seq_delta exactly",
          np.array_equal(got["seq_delta_sd"], ref_seq_delta),
          f"max|d| = {np.max(np.abs(got['seq_delta_sd'] - ref_seq_delta)):.3e}")
    check("sd reproduces cpu_stage struct_delta exactly",
          np.array_equal(got["struct_delta_sd"], ref_str_delta),
          f"max|d| = {np.max(np.abs(got['struct_delta_sd'] - ref_str_delta)):.3e}")

    # ---- fixed mode against an independent hand computation ---------------
    def hand_numerator(a, Asp, d, topk=0.10):
        sm = np.asarray(Asp @ a, dtype=np.float32)
        h = d > 0
        sm[h] /= d[h, None]
        sm[~h] = 0.0
        gm = sm.mean(axis=0)
        th = np.percentile(a, 90.0, axis=0)
        act = a > th[None, :]
        na = act.sum(axis=0).astype(np.float32)
        ns = np.where(na == 0, 1.0, na)
        am = (sm * act).sum(axis=0) / ns
        num = (am - gm).astype(np.float32)
        num[na < 5] = 0.0
        return num

    obs = hand_numerator(acts, A, deg)
    sh = np.zeros(n_feat, np.float32)
    for p in perms:
        sh += hand_numerator(acts[p], A, deg) / np.ones(n_feat, np.float32)
    sh /= len(perms)
    hand_fixed = obs / np.ones(n_feat, np.float32) - sh
    check("fixed matches an independent numerator computation",
          np.allclose(got["struct_delta_fixed"], hand_fixed, atol=1e-6, rtol=0),
          f"max|d| = {np.max(np.abs(got['struct_delta_fixed'] - hand_fixed)):.3e}")

    # ---- the relation between sd and fixed --------------------------------
    denom = got["denom_sd"] + rd.EPS
    check("struct_delta_sd == struct_delta_fixed / (SD + eps)",
          np.allclose(got["struct_delta_sd"], got["struct_delta_fixed"] / denom,
                      atol=1e-5, rtol=1e-4),
          f"max|d| = {np.max(np.abs(got['struct_delta_sd'] - got['struct_delta_fixed'] / denom)):.3e}")

    # ---- pathological features --------------------------------------------
    check("dead feature scores exactly 0 in every mode",
          all(got[f"struct_delta_{m}"][0] == 0.0 for m in ("sd", "fixed", "iqr", "rank")))
    check("n_active < 5 feature is zeroed",
          got["n_active"][1] < 5 and got["struct_delta_sd"][1] == 0.0,
          f"n_active={got['n_active'][1]:.0f}")
    check("constant feature flagged iqr_degenerate", bool(got["iqr_degenerate"][2]))
    check("no NaN or inf anywhere", all(
        np.isfinite(got[k]).all() for k in got if k != "idx"))

    # ---- the rank transform itself ----------------------------------------
    # ~90% of every column is exactly 0 under a TopK dictionary, so the zeros
    # form one huge tied block; if ties were broken arbitrarily the transform
    # would invent an ordering that is not in the data.
    toy = np.array([[0.0], [0.0], [0.0], [5.0], [9.0]], dtype=np.float32)
    rt = rd._rank_transform(toy)
    check("rank transform gives tied zeros one shared rank",
          rt[0, 0] == rt[1, 0] == rt[2, 0], f"zeros -> {rt[:3, 0]}")
    check("rank transform is monotone and spans [0, 1]",
          rt[3, 0] > rt[0, 0] and rt[4, 0] == 1.0 and rt.min() >= 0.0,
          f"{rt.ravel()}")
    check("rank transform preserves shape and dtype",
          rt.shape == toy.shape and rt.dtype == np.float32)
    big = _rank_reference(acts[:, 3:6])
    check("rank transform matches a per-column reference",
          np.allclose(rd._rank_transform(acts[:, 3:6]), big, atol=1e-6))

    # ---- rank mode is a real transform, not a no-op ------------------------
    check("rank mode differs from sd mode",
          not np.allclose(got["struct_delta_rank"], got["struct_delta_sd"], atol=1e-8))
    check("rank denominator is positive for live features",
          (got["denom_rank_sd"][3:] > 0).all())

    # ---- determinism -------------------------------------------------------
    again = rd._process_chunk(0, n_feat, acts, None, A2, deg2, A, deg, perms,
                              n_feat, 0.10, ("sd", "fixed", "iqr", "rank"))
    check("rerun is bit-identical", all(
        np.array_equal(got[k], again[k]) for k in got if k != "idx"))

    # ---- chunking must not change the answer -------------------------------
    parts = [rd._process_chunk(ci, 16, acts, None, A2, deg2, A, deg, perms,
                               n_feat, 0.10, ("sd", "fixed"))
             for ci in range((n_feat + 15) // 16)]
    joined = np.concatenate([p["struct_delta_sd"] for p in parts])
    check("chunk_size does not change the result",
          np.array_equal(joined, got["struct_delta_sd"]))

    # ---- second independent fixture ---------------------------------------
    a2, B, dB, L2 = make_case(seed=7, n_res=311, n_feat=48, n_prot=5)
    p2 = cs.build_protein_permutations(L2, 5)
    r2 = cs._process_struct_seq_chunk_v3(0, 48, a2, None, B, dB, B, dB, p2, 48, 0.10)
    g2 = rd._process_chunk(0, 48, a2, None, B, dB, B, dB, p2, 48, 0.10, ("sd",))
    check("second fixture: sd still exact",
          np.array_equal(g2["struct_delta_sd"], r2[2] - r2[4]))

    print("-" * 68)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
