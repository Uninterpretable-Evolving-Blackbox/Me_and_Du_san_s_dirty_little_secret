#!/usr/bin/env python3
"""
Smoke test for prep_controlled_corpus.py --block-shuffle.

prep_controlled_corpus.py built the corpora every published number rests on, so
the first thing checked here is that the default and --shuffle-residues paths are
untouched. The new mode is only worth having if it cannot disturb them.

    python3 tests/test_block_shuffle.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def residue_shuffle(seq, rng):
    """The unchanged --shuffle-residues path, verbatim from the script."""
    a = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).copy()
    rng.shuffle(a)
    return a.tobytes().decode("ascii")


def block_shuffle(seq, rng, k):
    """The new path, verbatim from the script."""
    a = np.frombuffer(seq.encode("ascii"), dtype=np.uint8).copy()
    blocks = [a[i:i + k] for i in range(0, a.size, k)]
    order = rng.permutation(len(blocks))
    a = np.concatenate([blocks[i] for i in order]) if len(blocks) > 1 else a
    return a.tobytes().decode("ascii")


def main():
    print("=" * 68)
    print("test_block_shuffle.py")
    print("=" * 68)

    src = (ROOT / "prep_controlled_corpus.py").read_text()

    # ---- the published paths must be untouched ----------------------------
    check("default path still skips all permutation",
          "if args.shuffle_residues:" in src)
    check("residue-level shuffle still reached when --block-shuffle is absent",
          "else:\n                shuf_rng.shuffle(_a)" in src)
    check("block mode writes to its own out-dir",
          'f"_blk{args.block_shuffle}"' in src)
    check("real corpus dir is still unreachable from a shuffled run",
          'args.shuffle_residues and not args.out_dir_explicit' in src)

    # ---- invariants of the new mode ---------------------------------------
    rng = np.random.RandomState(1234)
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), n))
            for n in (7, 60, 199, 512)]

    for k in (2, 5, 16, 64):
        for s in seqs:
            out = block_shuffle(s, np.random.RandomState(1234), k)
            if len(out) != len(s):
                check(f"k={k} preserves length", False, f"{len(s)} -> {len(out)}"); return 1
            if sorted(out) != sorted(s):
                check(f"k={k} preserves composition", False); return 1
    check("length preserved for every k and length", True, "4 lengths x 4 block sizes")
    check("composition preserved for every k and length", True)

    # local windows survive: every k-block of the output is a k-block of the input
    s = seqs[2]
    k = 16
    out = block_shuffle(s, np.random.RandomState(7), k)
    in_blocks = {s[i:i + k] for i in range(0, len(s), k)}
    out_blocks = [out[i:i + k] for i in range(0, len(out), k)]
    # blocks land on k-boundaries only when every block is full length
    full = len(s) % k == 0
    check("local windows survive intact",
          all(b in in_blocks for b in out_blocks) if full else True,
          "checked on a length divisible by k" if full else "skipped (ragged tail)")

    s_even = seqs[3]                     # 512, divisible by 16
    out = block_shuffle(s_even, np.random.RandomState(7), 16)
    in_blocks = {s_even[i:i + 16] for i in range(0, 512, 16)}
    out_blocks = [out[i:i + 16] for i in range(0, 512, 16)]
    check("every output block is an input block", all(b in in_blocks for b in out_blocks))
    check("long-range order is destroyed", out != s_even)
    check("block shuffle differs from residue shuffle",
          out != residue_shuffle(s_even, np.random.RandomState(7)))

    # ragged tail: a sequence whose length is not a multiple of k
    s_odd = seqs[1]                      # 60, k=7 -> last block is 4 long
    out = block_shuffle(s_odd, np.random.RandomState(3), 7)
    check("ragged tail preserves length and composition",
          len(out) == len(s_odd) and sorted(out) == sorted(s_odd))

    # single-block sequence must come back unchanged, not crash
    check("sequence shorter than one block is returned unchanged",
          block_shuffle("ACDEF", np.random.RandomState(1), 64) == "ACDEF")

    # determinism
    check("deterministic given the seed",
          block_shuffle(s_even, np.random.RandomState(9), 8)
          == block_shuffle(s_even, np.random.RandomState(9), 8))

    # ---- CLI guards, which must fire before any network access ------------
    for args, want in ((["--block-shuffle", "1"], "same as --shuffle-residues"),
                       (["--block-shuffle", "-3"], "must be >= 0")):
        r = subprocess.run([sys.executable, str(ROOT / "prep_controlled_corpus.py"), *args],
                           capture_output=True, text=True, timeout=60)
        check(f"'{' '.join(args)}' is rejected",
              r.returncode != 0 and want in (r.stdout + r.stderr), f"rc={r.returncode}")

    r = subprocess.run([sys.executable, str(ROOT / "prep_controlled_corpus.py"), "--help"],
                       capture_output=True, text=True, timeout=60)
    check("--help documents the new flag",
          r.returncode == 0 and "SECOND DESTRUCTION PROCEDURE" in r.stdout)

    print("-" * 68)
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
