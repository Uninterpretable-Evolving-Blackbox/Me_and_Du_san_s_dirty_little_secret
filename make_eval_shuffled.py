#!/usr/bin/env python3
"""
make_eval_shuffled.py -- write an evaluation set with residue order permuted.

Copies an existing eval set and permutes the residues inside each sequence.
Everything else is carried across unchanged:

    permuted   the residue string in sequences.json
    unchanged  uids.json, META.json, and therefore the validation split

Sequence lengths and per-protein amino-acid composition are preserved exactly,
since a permutation reorders the same multiset. Downstream scoring reads
coordinates from cache/pdb_files by position and is not touched by this script.

USAGE
    python make_eval_shuffled.py                      # eval_set -> eval_set_evalshuf
    python make_eval_shuffled.py --seed 43 --out eval_set_evalshuf_s43

Deterministic given --seed. Re-running overwrites the output directory.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set",
                    help="source eval set (needs uids.json, sequences.json, META.json)")
    ap.add_argument("--out", default="eval_set_evalshuf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.eval_set), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"no such eval set: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    uids = json.loads((src / "uids.json").read_text())
    seqs = json.loads((src / "sequences.json").read_text())
    meta = json.loads((src / "META.json").read_text())

    rng = random.Random(args.seed)

    def shuffled(s):
        chars = list(s)
        rng.shuffle(chars)
        return "".join(chars)

    if isinstance(seqs, dict):
        out_seqs = {k: shuffled(v) for k, v in seqs.items()}
        n, lengths_ok = len(out_seqs), all(
            len(out_seqs[k]) == len(seqs[k]) for k in seqs)
    else:
        out_seqs = [shuffled(s) for s in seqs]
        n, lengths_ok = len(out_seqs), all(
            len(a) == len(b) for a, b in zip(out_seqs, seqs))

    if not lengths_ok:
        raise SystemExit("length changed during permutation -- refusing to write")

    meta = dict(meta)
    meta["eval_order_destroyed"] = True
    meta["eval_shuffle_seed"] = args.seed
    meta["derived_from"] = str(src)
    meta["note"] = ("Residue order permuted within each sequence. uids, lengths "
                    "and val split unchanged.")

    (dst / "uids.json").write_text(json.dumps(uids))
    (dst / "sequences.json").write_text(json.dumps(out_seqs))
    (dst / "META.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {dst}/  ({n} sequences, seed {args.seed})")
    print("  uids, lengths and val split preserved; residue order permuted")


if __name__ == "__main__":
    main()
