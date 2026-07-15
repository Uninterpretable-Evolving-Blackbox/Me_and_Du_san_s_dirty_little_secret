#!/usr/bin/env python3
"""
prep_controlled_corpus.py — build ONE shared tokenised corpus for the controlled
MLM-vs-CLM experiment. Both objectives train on THIS exact corpus (same sequences,
same order); only the objective differs. That is what isolates the data.

Anchored to ESM-C: tokenised with EvolutionaryScale's own EsmSequenceTokenizer
(residue ids 4..23 contiguous, cls=0, pad=1, eos=2, mask=32), so tokenisation matches
the architecture's reference instead of a homemade vocab.

- streams ConvergeBio/uniref50 (cc-by-4.0) from HuggingFace (no full download)
- keeps standard-20-AA sequences, length >= min_len, truncates to context-2
- holds out sequences that EXACTLY match a SCOPe eval domain
- writes flat uint8 tokens + lengths + offsets

Deterministic: the HF stream is shuffled with a fixed seed (42) and fixed buffer, so
re-running reproduces the corpus. cache/scope_40.fa MUST be present, else the eval
holdout silently differs.

Usage:
  python prep_controlled_corpus.py --smoke     # 2k seqs, quick end-to-end test
  python prep_controlled_corpus.py            # 3M seqs (~1 GB, 30-60 min)
"""
import argparse
import json
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)


def esmc_vocab():
    """Residue->id map and special ids, taken from ESM-C's own tokenizer."""
    from esm.tokenization import get_esmc_model_tokenizers

    t = get_esmc_model_tokenizers()
    aa2id = {a: int(t.encode(a)[1]) for a in AA}
    lo, hi = min(aa2id.values()), max(aa2id.values())
    assert sorted(aa2id.values()) == list(range(lo, hi + 1)), "residue ids not contiguous"
    return aa2id, dict(pad=int(t.pad_token_id), bos=int(t.cls_token_id),
                       eos=int(t.eos_token_id), mask=int(t.mask_token_id),
                       aa_lo=lo, aa_hi=hi, tokenizer_vocab_size=int(t.vocab_size))


def load_scope_seqs(fasta):
    p = Path(fasta)
    if not p.exists():
        raise SystemExit(
            f"ERROR: SCOPe fasta not found at {fasta}\n"
            "It ships with this repo (cache/scope_40.fa). Without it the eval-domain "
            "holdout is skipped and the corpus will NOT match the reference run."
        )
    seqs, cur = set(), []
    with open(p) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.add("".join(cur)); cur = []
            else:
                cur.append(line.strip().upper())
    if cur:
        seqs.add("".join(cur))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=3_000_000)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--hf-dataset", default="ConvergeBio/uniref50")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scope-fasta", default="cache/scope_40.fa")
    ap.add_argument("--out-dir", default=str(Path.home() / "own_sae_data" / "uniref50_pilot"))
    ap.add_argument("--shuffle-buffer", type=int, default=50000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_sequences, args.shuffle_buffer = 2000, 5000
        args.out_dir = str(Path.home() / "own_sae_data" / "uniref50_smoke")

    from datasets import load_dataset

    aa2id, sp = esmc_vocab()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    maxres = args.context - 2

    scope = load_scope_seqs(args.scope_fasta)
    print(f"SCOPe holdout sequences: {len(scope)}")
    print(f"ESM-C tokenizer: residues {sp['aa_lo']}..{sp['aa_hi']} | "
          f"pad={sp['pad']} bos={sp['bos']} eos={sp['eos']} mask={sp['mask']}")

    ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=42, buffer_size=args.shuffle_buffer)
        print(f"streaming shuffle buffer: {args.shuffle_buffer} (seed 42)")

    tokens, lengths = [], []
    kept = seen = d_aa = d_len = d_hold = 0
    seq_col = None
    for rec in ds:
        seen += 1
        if seq_col is None:
            for c in ("sequence", "Sequence", "text", "seq", "Seq"):
                if c in rec:
                    seq_col = c; break
            if seq_col is None:
                seq_col = [k for k, v in rec.items() if isinstance(v, str)][0]
            print(f"using sequence column: '{seq_col}'")
        s = str(rec[seq_col]).strip().upper()
        if len(s) > maxres:
            s = s[:maxres]
        if len(s) < args.min_len:
            d_len += 1; continue
        if not set(s) <= AA_SET:
            d_aa += 1; continue
        if s in scope:
            d_hold += 1; continue
        ids = [sp["bos"]] + [aa2id[a] for a in s] + [sp["eos"]]
        tokens.append(np.array(ids, dtype=np.uint8))
        lengths.append(len(ids))
        kept += 1
        if kept >= args.n_sequences:
            break
        if kept % 200000 == 0:
            print(f"  kept {kept} / seen {seen}")

    flat = np.concatenate(tokens)
    lengths = np.array(lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    np.save(out / "tokens.npy", flat)
    np.save(out / "lengths.npy", lengths)
    np.save(out / "offsets.npy", offsets)
    meta = dict(aa2id=aa2id, context=args.context, n_sequences=int(kept),
                n_tokens=int(flat.shape[0]), hf_dataset=args.hf_dataset, seq_col=seq_col,
                dropped=dict(aa=d_aa, length=d_len, holdout=d_hold),
                tokenizer="esm-c EsmSequenceTokenizer",
                vocab_size=64,  # ESM-C's embedding slot count: nn.Embedding(64, d_model)
                **sp)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDONE: kept={kept} seqs | {flat.shape[0]} tokens (~{flat.shape[0]/1e6:.1f}M) | "
          f"mean_len={lengths.mean():.1f}")
    print(f"dropped: non-std-aa={d_aa}  too-short={d_len}  scope-holdout={d_hold}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
