#!/usr/bin/env python3
"""
aa_floor.py — the residue-type floor for InterPLM's REPORTED concepts.

WHAT IT ASKS
------------
InterPLM's report_metrics.py drops `amino_acid_*` from headline numbers, with the
comment that a single token "does not necessarily represent a biological
concept". Correct -- but the filter is a substring match, and it leaves in
concepts that are subsets of one residue type by construction:

    Modified residue_Phosphoserine   ⊂ serine
    Disulfide bond                   = cysteine
    Glycosylation_N-linked…asparagine ⊂ asparagine

For those, a detector that knows nothing but "this residue is a serine" already
achieves some F1. That is a floor, and concept-F1 reports no floor.

WHAT IT COMPUTES
----------------
Per reported concept: the best F1 obtainable from a single `amino_acid_X` column,
under InterPLM's own scoring, read off their own eval matrices --

    precision         = tp / (tp + fp)                     per residue
    recall            = tp / n_positive_aa                 per residue
    recall_per_domain = tp_domains / n_positive_domains    per domain
    f1_per_domain     = harmonic(precision, recall_per_domain)

with recall_per_domain set equal to recall for AA-level concepts, exactly as
calculate_f1.py does. Domain identity comes from the matrix VALUES, which store
domain ids -- the same quantity their count_unique_nonzero_sparse uses.

SELECTION MATCHES THEIRS. The best amino acid per concept is chosen on VALID and
scored on TEST, mirroring report_metrics choosing the best feature on valid and
reporting on test. Selecting in-sample would flatter the floor.

No model, no SAE, no embeddings: this is a property of their annotation set, so
it is independent of whatever model is later compared against it.

USAGE
  python aa_floor.py --valid <ann>/valid --test <ann>/test [--sae-pairings <heldout_top_pairings.csv>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _imports(repo):
    sys.path.insert(0, repo)
    from interplm.analysis.concepts.concept_constants import is_aa_level_concept
    from interplm.analysis.concepts.report_metrics import concept_types_to_ignore
    return is_aa_level_concept, concept_types_to_ignore


def load_matrix(eval_dir: Path):
    meta = json.loads((eval_dir / "metadata.json").read_text())
    names = [n for n in (eval_dir / "aa_concepts_columns.txt").read_text().split("\n") if n]
    keep = np.asarray(meta["indices_of_concepts_to_keep"], dtype=int)
    mats = [sp.load_npz(meta["path_to_shards"][s]).tocsc()[:, keep]
            for s in sorted(meta["path_to_shards"], key=int)]
    M = sp.vstack(mats).tocsc()
    assert M.shape[1] == len(names), (M.shape, len(names))
    return M, names


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def score_split(M, names, is_aa_level, ignore):
    """{concept: {aa_letter: f1_per_domain}} for every reported concept."""
    aa_cols = {n[-1]: i for i, n in enumerate(names) if n.startswith("amino_acid_")}
    letters = sorted(aa_cols)
    n_res = M.shape[0]

    # dense per-residue amino-acid code (12M int8 = 12 MB), -1 = none
    code = np.full(n_res, -1, dtype=np.int8)
    for k, L in enumerate(letters):
        code[M[:, aa_cols[L]].indices] = k
    aa_total = np.bincount(code[code >= 0], minlength=len(letters)).astype(np.float64)

    out = {}
    for ci, cname in enumerate(names):
        if any(t.lower() in cname.lower() for t in ignore):
            continue
        col = M[:, ci]
        rows, doms = col.indices, col.data
        if rows.size == 0:
            continue
        codes = code[rows]
        valid = codes >= 0
        if not valid.any():
            continue
        tp = np.bincount(codes[valid], minlength=len(letters)).astype(np.float64)
        n_pos_aa = float(rows.size)
        n_pos_dom = float(np.unique(doms).size)
        aa_level = is_aa_level(cname)

        per_aa = {}
        for k, L in enumerate(letters):
            if tp[k] == 0:
                per_aa[L] = 0.0
                continue
            precision = tp[k] / aa_total[k]
            recall = tp[k] / n_pos_aa
            if aa_level:
                rec = recall
            else:
                sel = valid & (codes == k)
                rec = np.unique(doms[sel]).size / n_pos_dom if n_pos_dom else 0.0
            per_aa[L] = f1(precision, rec)
        out[cname] = per_aa
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--repo", default=str(Path.home() /
                    "own_sae_data/interplm_stress/interplm_repo"))
    ap.add_argument("--sae-pairings", default=None,
                    help="their heldout_top_pairings.csv on the test split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    is_aa_level, ignore = _imports(args.repo)
    Mv, nv = load_matrix(Path(args.valid))
    Mt, nt = load_matrix(Path(args.test))
    assert nv == nt, "valid/test concept columns differ"
    print(f"  valid {Mv.shape[0]:,} residues | test {Mt.shape[0]:,} | {len(nv)} concepts")

    sv = score_split(Mv, nv, is_aa_level, ignore)
    st = score_split(Mt, nt, is_aa_level, ignore)

    rows = []
    for c, per_aa in sv.items():
        if c not in st:
            continue
        best = max(per_aa, key=per_aa.get)            # chosen on VALID
        rows.append({"concept": c, "aa_level": is_aa_level(c),
                     "best_aa": best,
                     "aa_valid_f1": per_aa[best],
                     "aa_test_f1": st[c][best]})       # reported on TEST
    df = pd.DataFrame(rows)

    if args.sae_pairings and Path(args.sae_pairings).exists():
        sae = pd.read_csv(args.sae_pairings)
        cols = [c for c in ("concept", "feature", "f1_per_domain") if c in sae.columns]
        sae = sae[cols].rename(columns={"f1_per_domain": "sae_test_f1",
                                        "feature": "sae_feature"})
        df = df.merge(sae, on="concept", how="left")
        df["sae_minus_aa"] = df["sae_test_f1"] - df["aa_test_f1"]
        m = df.dropna(subset=["sae_test_f1"])
        if len(m):
            df["frac_reachable"] = df["aa_test_f1"] / df["sae_test_f1"]

    df = df.sort_values("aa_test_f1", ascending=False)
    df.to_csv(args.out, index=False)

    print()
    print("  HIGHEST RESIDUE-TYPE FLOORS (held out: aa chosen on valid, scored on test)")
    print("  %-52s %4s %8s" % ("concept", "aa", "F1"))
    for _, r in df.head(20).iterrows():
        print("  %-52s %4s %8.3f" % (r.concept[:52], r.best_aa, r.aa_test_f1))
    print()
    print("  concepts: %d | floor >0.5: %d | >0.3: %d | >0.1: %d | median %.3f"
          % (len(df), (df.aa_test_f1 > .5).sum(), (df.aa_test_f1 > .3).sum(),
             (df.aa_test_f1 > .1).sum(), df.aa_test_f1.median()))
    print(f"\n  wrote -> {args.out}")


if __name__ == "__main__":
    main()
