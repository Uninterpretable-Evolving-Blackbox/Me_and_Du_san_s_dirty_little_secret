#!/usr/bin/env python3
"""make_folddisjoint_eval_set.py — build a fold-disjoint SAE/probe split, and price the old one.

The asymmetry a reviewer will find: concept-F1 already uses a FOLD-DISJOINT val/test split
(`experiment_concept_f1.py --split-level fold`), while the 1,350/150 split that trains the
dictionaries and the probes is a UNIFORM RANDOM partition over domains. Only one of the two
is homology-aware, and there is no principled reason for that.

`eval_ctrl_plm.py` reads the split from `<eval-set>/META.json:val_uids` and nothing else
(`load_eval_set`, eval_ctrl_plm.py:86-96), so making it fold-disjoint needs no code change
and no retraining of any pLM — only a new eval-set directory and a re-run of the SAE fit.

This script does two things:

  1. REPORTS how much fold leakage the current split actually has: how many of the 150
     held-out domains share a SCOPe fold with a domain in the 1,350 used for fitting. If the
     answer is near zero the asymmetry is cosmetic and can be documented instead of fixed --
     that is a cheaper outcome and worth knowing before spending GPU time.

  2. WRITES `eval_set_folddisj/` with the same uids and sequences and a val_uids chosen so
     that no SCOPe fold appears on both sides.

The fold assignment is not reimplemented here: it reuses `split_proteins` from
`experiment_concept_f1.py`, which derives folds from the SCOPe sccs codes in the FASTA, so
both splits in the paper come from one definition of "fold".

Usage:
    python make_folddisjoint_eval_set.py --report-only        # just price the current split
    python make_folddisjoint_eval_set.py                      # also write eval_set_folddisj/
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def load_folds(uids, fasta_path):
    """uid -> fold string, from the one definition of "fold" the project already uses.

    `cluster_bootstrap.load_uid_clusters` is what `split_proteins` calls internally and what
    the L_struct cluster bootstrap resamples over, so taking it from here keeps the new split,
    the concept-F1 split and the confidence intervals on a single fold assignment.
    Domains missing from the FASTA become their own singleton cluster, which is the
    conservative choice: an unknown domain is never merged with another.
    """
    try:
        from cluster_bootstrap import load_uid_clusters
    except ImportError as e:
        raise SystemExit(f"cannot import load_uid_clusters from cluster_bootstrap.py: {e}")
    cl = load_uid_clusters(fasta_path, level="fold")
    return {str(u): cl.get(str(u), f"__singleton__{u}") for u in uids}


def report(uids, val_uids, folds):
    train = [u for u in uids if u not in val_uids]
    val = [u for u in uids if u in val_uids]
    tr_folds = {folds[u] for u in train}
    leaked = [u for u in val if folds[u] in tr_folds]
    print(f"  domains            : {len(uids)}  ({len(train)} fit / {len(val)} held out)")
    print(f"  distinct folds     : {len({folds[u] for u in uids})}")
    print(f"  held-out domains sharing a fold with the fit set: "
          f"{len(leaked)} / {len(val)}  ({100.0*len(leaked)/max(1,len(val)):.1f}%)")
    return leaked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--out", default="eval_set_folddisj")
    ap.add_argument("--fasta-path", default="cache/scope_40.fa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    src = Path(a.eval_set)
    uids = [str(u) for u in json.loads((src / "uids.json").read_text())]
    meta = json.loads((src / "META.json").read_text())
    val_uids = {str(u) for u in meta.get("val_uids", [])}
    if not Path(a.fasta_path).exists():
        raise SystemExit(f"no SCOPe FASTA at {a.fasta_path} — cannot assign folds")

    folds = load_folds(uids, a.fasta_path)

    print("CURRENT split (uniform random over domains):")
    leaked = report(uids, val_uids, folds)
    if not leaked:
        print("\n  The current split is already fold-disjoint. Document it and stop —")
        print("  there is nothing here worth GPU time.")
        return 0

    if a.report_only:
        print("\n  --report-only: nothing written.")
        return 0

    # val_frac from the existing split so the new one is the same size
    from experiment_concept_f1 import split_proteins
    val_frac = len(val_uids) / len(uids)
    new_val, _ = split_proteins(uids, seed=a.seed, val_frac=val_frac,
                                fasta_path=a.fasta_path, level="fold")
    new_val = {str(u) for u in new_val}

    print(f"\nNEW split (fold-disjoint, seed {a.seed}, val_frac {val_frac:.3f}):")
    residual = report(uids, new_val, folds)
    if residual:
        print(f"  WARNING: {len(residual)} still share a fold — split_proteins fell back to a")
        print("  protein-level split. Check the FASTA path and the sccs codes before using this.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("uids.json", "sequences.json"):
        shutil.copy2(src / f, out / f)
    new_meta = dict(meta)
    new_meta["val_uids"] = sorted(new_val)
    new_meta["split_level"] = "fold"
    new_meta["split_seed"] = a.seed
    new_meta["note"] = (f"Fold-disjoint val split built by make_folddisjoint_eval_set.py from "
                        f"{src}/META.json. No SCOPe fold appears in both the fit and held-out "
                        f"sets. Sizes match the original split.")
    (out / "META.json").write_text(json.dumps(new_meta, indent=1))

    n_moved = len(new_val ^ val_uids) // 2
    print(f"\n  wrote {out}/  ({len(new_val)} held out; ~{n_moved} domains differ from the old split)")
    print(f"  use it with:  python eval_ctrl_plm.py --eval-set {out} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
