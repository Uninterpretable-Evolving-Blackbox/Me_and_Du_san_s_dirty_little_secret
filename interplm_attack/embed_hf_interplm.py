#!/usr/bin/env python3
"""
embed_hf_interplm.py — write a PUBLISHED HuggingFace pLM's activations in
InterPLM's input format, so their metric runs on it with nothing modified.

Sibling of embed_ctrl_interplm.py, which does the same for our own checkpoints.
Same output contract, same alignment guard, same .done semantics.

WHY
---
Every published pLM SAE paper -- InterPLM, Adams et al., the Matryoshka ESM2-3B
work, Villegas Garcia & Ansuini, Nainani et al. -- evaluates on ESM-2, which is
masked. Not one uses a causal pLM. RITA_l is causal, published, and an almost
exact size match for ESM-2-650M (680M vs 650M), which makes the pair the closest
thing to a controlled objective comparison that exists among released models.

So: run InterPLM's own concept-F1 on both arms of that pair. If the metric
behaves differently on the causal arm, the field's untested scope assumption is
wrong on a model anyone can download -- not only on our 42M pair.

OUTPUT (identical to embed_ctrl_interplm.py)
--------------------------------------------
  --shards-dir -> <out>/shard_N/embeddings.pt {embeddings, boundaries, protein_ids}
  --fasta-dir  -> <out>/shard_N/activations.pt + metadata.yaml
  .done sentinel written only after every shard succeeds

MODEL NOTES
-----------
RITA's custom modeling code needs three workarounds, all of them established the
expensive way in extract_embeddings.py and reproduced here verbatim in intent:

 1. It predates transformers 5.x's tied-weights API, so PreTrainedModel needs a
    default `all_tied_weights_keys` or the load path raises.
 2. It must be loaded in fp32. RITA_l ships fp16, and transformers 5.x honours
    the checkpoint dtype -- but RITA's attention mixes an fp32-upcast softmax
    with fp16 value projections, which raises a dtype error on CPU and silently
    produces NaN in deep blocks on MPS (observed at layer 12).
 3. Its forward ignores output_hidden_states and returns only the final state,
    so per-block activations must come from forward hooks on
    model.transformer.layers[i].

Setting tokenizer.eos_token is specifically NOT done: it mutates tokenizer state
in a way that propagates into attention-mask handling and produces NaN in deep
blocks. Only pad_token is set.

Because of (2) and (3) this script hard-checks for NaN/Inf before writing
anything. A silent NaN here would flow into the SAE and out the other side as a
plausible concept-F1 number.

USAGE
-----
  python embed_hf_interplm.py --model rita_l --layer 13 --smoke
  python embed_hf_interplm.py --model rita_l --layer 13 \\
      --shards-dir $BASE/ann/processed --out $BASE/embd_analysis/rita_l/L13
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# name -> (hf id, n_blocks, family). n_blocks is asserted after load, not trusted.
MODELS = {
    "rita_s":     ("lightonai/RITA_s",                  12, "rita"),
    "rita_l":     ("lightonai/RITA_l",                  24, "rita"),
    "esm2_8m":    ("facebook/esm2_t6_8M_UR50D",          6, "esm"),
    "esm2_650m":  ("facebook/esm2_t33_650M_UR50D",      33, "esm"),
}

_WARN = {"nonstd": 0, "trunc": 0}


# --------------------------------------------------------------------------
def load_model(name, device):
    hf_id, n_blocks, family = MODELS[name]
    from transformers import AutoTokenizer

    if family == "rita":
        from transformers import AutoModelForCausalLM
        # (1) tied-weights shim for transformers 5.x
        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        # (2) fp32 explicitly -- see module docstring
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=torch.float32)
        model = model.float().to(device).eval()
        if tok.pad_token is None:
            vocab = tok.get_vocab()
            tok.pad_token = "<PAD>" if "<PAD>" in vocab else tok.convert_ids_to_tokens(1)
        blocks = getattr(getattr(model, "transformer", None), "layers", None)
        if blocks is None:
            raise SystemExit("RITA load path changed: expected model.transformer.layers")
        n_real = len(blocks)
    else:
        from transformers import EsmModel
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = EsmModel.from_pretrained(hf_id).to(device).eval()
        n_real = model.config.num_hidden_layers
        blocks = None

    if n_real != n_blocks:
        print(f"  ! block count {n_real} != expected {n_blocks}; using {n_real}")
    return model, tok, blocks, n_real, family, hf_id


def check_1to1(tok, family, probe="MKVLWAFTGVVPILVELDGDVNGHKF"):
    """One token per residue, or every downstream row index is wrong."""
    if family == "rita":
        ids = tok(probe, add_special_tokens=False)["input_ids"]
    else:
        ids = tok(probe, add_special_tokens=True)["input_ids"][1:-1]
    if len(ids) != len(probe):
        raise SystemExit(
            f"  TOKENISER NOT 1:1 — {len(ids)} tokens for a {len(probe)}-residue "
            f"probe. Every residue index downstream would be wrong. Refusing.")
    return True


# --------------------------------------------------------------------------
def embed(model, tok, blocks, family, seqs, layer, device, bs, max_len):
    feats, bounds, cur = [], [], 0
    grab = {}

    if family == "rita":
        def hook(_m, _i, o):
            grab["h"] = (o[0] if isinstance(o, tuple) else o).detach()
        handle = blocks[layer].register_forward_hook(hook)

    try:
        with torch.no_grad():
            for i in range(0, len(seqs), bs):
                chunk = [s.upper()[:max_len] for s in seqs[i:i + bs]]
                for s, orig in zip(chunk, seqs[i:i + bs]):
                    if len(orig) > max_len:
                        _WARN["trunc"] += 1
                if family == "rita":
                    enc = tok(chunk, return_tensors="pt", padding=True,
                              add_special_tokens=False)
                    model(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device))
                    h = grab["h"].float().cpu().numpy()
                    starts = [0] * len(chunk)          # no BOS
                else:
                    enc = tok(chunk, return_tensors="pt", padding=True)
                    out = model(input_ids=enc["input_ids"].to(device),
                                attention_mask=enc["attention_mask"].to(device),
                                output_hidden_states=True)
                    h = out.hidden_states[layer].float().cpu().numpy()
                    starts = [1] * len(chunk)          # strip <cls>
                for r, s in enumerate(chunk):
                    L = len(s)
                    feats.append(h[r, starts[r]:starts[r] + L, :])
                    bounds.append((cur, cur + L)); cur += L
    finally:
        if family == "rita":
            handle.remove()

    X = np.concatenate(feats, 0).astype(np.float32)
    # RITA on MPS has produced NaN in deep blocks under the wrong dtype. A NaN
    # here would pass straight through the SAE and out as a plausible F1.
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad:
        raise SystemExit(
            f"  {n_bad:,} non-finite values in the activations. Refusing to "
            f"write. Try --device cpu; this is the documented RITA/MPS failure.")
    return X, bounds


def meta_yaml(d, n, dim, layer, tag):
    (d / "metadata.yaml").write_text(
        f"d_model: {dim}\ndtype: float32\nlayer: {layer}\nmodel: {tag}\n"
        f"total_tokens: {n}\n")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--fasta-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="embed only the first N shards (for a timed pilot)")
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    model, tok, blocks, n_blocks, family, hf_id = load_model(args.model, args.device)
    check_1to1(tok, family)
    dim = (model.config.d_model if family == "rita" and hasattr(model.config, "d_model")
           else getattr(model.config, "hidden_size", None))
    tag = args.model_tag or args.model
    print(f"  {hf_id}  family={family}  blocks={n_blocks}  d_model={dim}  "
          f"device={args.device}  loaded in {time.time()-t0:.0f}s")
    if not (0 <= args.layer < n_blocks + (1 if family == "esm" else 0)):
        raise SystemExit(f"  --layer {args.layer} out of range for {n_blocks} blocks")

    if args.smoke:
        demo = ["MKTAYIAKQRQISFVKSHFSRQ", "MSKGEELFTGVVPILVELDGDV",
                "MVLSPADKTNVKAAWGKVGAHA", "MEEPQSDPSVEPPLSQETFSDL",
                "MADQLTEEQIAEFKEAFSLFDK", "MGSSHHHHHHSSGLVPRGSHM",
                "MTEYKLVVVGAGGVGKSALTI", "MSDNGPQNQRNAPRITFGGPS"]
        t = time.time()
        X, b = embed(model, tok, blocks, family, demo, args.layer,
                     args.device, 4, args.max_len)
        exp = sum(len(s) for s in demo)
        assert X.shape[0] == exp, f"ROW MISMATCH {X.shape[0]} vs {exp}"
        print(f"  SMOKE OK: {X.shape} rows, {len(b)} proteins, sum(len)={exp} "
              f"(equal), finite, {time.time()-t:.1f}s")
        return

    if not args.out:
        raise SystemExit("  --out required unless --smoke")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.fasta_dir:
        fastas = sorted(Path(args.fasta_dir).glob("shard_*.fasta"))
        if args.limit_shards:
            fastas = fastas[:args.limit_shards]
        if not fastas:
            raise SystemExit(f"  no shard_*.fasta under {args.fasta_dir}")
        for ff in fastas:
            seqs, cur = [], []
            for line in open(ff):
                if line.startswith(">"):
                    if cur: seqs.append("".join(cur)); cur = []
                else: cur.append(line.strip())
            if cur: seqs.append("".join(cur))
            t = time.time()
            X, _ = embed(model, tok, blocks, family, seqs, args.layer,
                         args.device, args.batch_size, args.max_len)
            sd = out / ff.stem; sd.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(X), sd / "activations.pt")
            meta_yaml(sd, X.shape[0], X.shape[1], args.layer, tag)
            print(f"  {ff.stem}: {X.shape[0]:,} x {X.shape[1]}  [{time.time()-t:.0f}s]")
        if _WARN["trunc"]:
            print(f"  !! {_WARN['trunc']} sequences truncated at {args.max_len}")
        if not args.limit_shards:
            (out / ".done").write_text(str(len(fastas)))
        return

    shard_files = sorted(Path(args.shards_dir).glob("shard_*/protein_data.tsv"))
    if args.limit_shards:
        shard_files = shard_files[:args.limit_shards]
    if not shard_files:
        raise SystemExit(f"  no shard_*/protein_data.tsv under {args.shards_dir}")
    for sf in shard_files:
        df = pd.read_csv(sf, sep="\t")
        col = next(c for c in df.columns if c.lower() == "sequence")
        expected = sp.load_npz(sf.parent / "aa_concepts.npz").shape[0]
        t = time.time()
        X, bounds = embed(model, tok, blocks, family, df[col].astype(str).tolist(),
                          args.layer, args.device, args.batch_size, args.max_len)
        # The alignment guard. extract_annotations does not truncate, so if any
        # sequence exceeded --max-len the concept matrix and the activation rows
        # are offset against each other and every concept score is meaningless.
        if X.shape[0] != expected:
            raise SystemExit(
                f"  {sf.parent.name}: ROW MISALIGNMENT — {X.shape[0]} activation "
                f"rows vs {expected} in aa_concepts.npz. Refusing to write.")
        sd = out / sf.parent.name; sd.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": torch.from_numpy(X), "boundaries": bounds,
                    "protein_ids": df["Entry"].tolist() if "Entry" in df.columns
                    else list(range(len(df)))}, sd / "embeddings.pt")
        meta_yaml(sd, X.shape[0], X.shape[1], args.layer, tag)
        print(f"  {sf.parent.name}: {X.shape[0]:,} x {X.shape[1]}  aligned  "
              f"[{time.time()-t:.0f}s]")
    if not args.limit_shards:
        (out / ".done").write_text(str(len(shard_files)))
    print(f"  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
