#!/usr/bin/env python3
"""
eval_ctrl_plm.py — turn a trained controlled PLM checkpoint into a "layer dir" that the
existing L_struct pipeline (cpu_stage.py) consumes unchanged.

Steps (mirrors run_unsupervised.py, with our anchored ESM-C PLM as the extractor):
  1. load checkpoint -> CtrlESMC
  2. reuse the SAME 1,500 eval proteins as ESM-2/RITA (eval_set/uids+sequences)
  3. extract per-residue hidden states at a given block index
  4. protein-split train/val (reuse the reference val_uids -> identical split)
  5. Bricken norm_scale -> train_sae (expansion 8, k=256) -> extract Z
  6. write Z.npy / D.npy / sae_model.pt / META.json / lengths / offsets / sequences / uids

Then: cpu_stage.py --layer-dir <out> --model-type residue
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_ctrl_esmc import CtrlESMC
from train_sae import compute_norm_scale, train_sae, extract_sae_features


def pick_device(pref):
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_eval_set(d):
    """uid -> sequence, plus the reference val split. No cpu_stage import (avoids umap)."""
    d = Path(d)
    uids = [str(u) for u in json.loads((d / "uids.json").read_text())]
    seqs = json.loads((d / "sequences.json").read_text())
    if isinstance(seqs, dict):
        uid2seq = {str(k): v for k, v in seqs.items()}
    else:
        uid2seq = {u: s for u, s in zip(uids, seqs)}
    meta = json.loads((d / "META.json").read_text())
    return uids, uid2seq, set(str(u) for u in meta.get("val_uids", []))


@torch.no_grad()
def extract_layer(model, uids, seqs, layer, aa2id, bos, eos, pad, device,
                  batch_size=16, max_len=512):
    """(sum_res, D) per-residue hidden states at block `layer`, and per-protein lengths."""
    model.eval()
    feats, lengths = [], []
    for i in range(0, len(uids), batch_size):
        chunk = list(range(i, min(i + batch_size, len(uids))))
        toks = [([bos] + [aa2id.get(a, aa2id["A"]) for a in seqs[j].upper()] + [eos])[:max_len]
                for j in chunk]
        T = max(len(t) for t in toks)
        ids = np.full((len(toks), T), pad, dtype=np.int64)
        am = np.zeros((len(toks), T), dtype=np.int64)
        for r, t in enumerate(toks):
            ids[r, :len(t)] = t
            am[r, :len(t)] = 1
        _, hid = model(torch.from_numpy(ids).to(device),
                       torch.from_numpy(am).to(device), return_hidden=True)
        h = hid[layer].float().cpu().numpy()
        for r, t in enumerate(toks):
            L = len(t) - 2                      # residues between BOS/EOS
            feats.append(h[r, 1:1 + L, :])
            lengths.append(L)
    return np.concatenate(feats, axis=0).astype(np.float32), np.array(lengths, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="e.g. ctrl_mlm_s42")
    ap.add_argument("--eval-set", default="eval_set",
                    help="dir with uids.json/sequences.json/META.json (the SAME 1500 proteins)")
    ap.add_argument("--layer", type=int, required=True, help="block index (0..n_layers-1)")
    ap.add_argument("--out-root", default="outputs_ctrl")
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sae-device", default="auto")
    ap.add_argument("--sae-epochs", type=int, default=60)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k-sparse", type=int, default=256)
    args = ap.parse_args()

    dev, sdev = pick_device(args.device), pick_device(args.sae_device)
    uids, uid2seq, val_uids = load_eval_set(args.eval_set)
    uids = [u for u in uids if u in uid2seq]
    if args.max_proteins and args.max_proteins < len(uids):
        uids = uids[:args.max_proteins]
    seqs = [uid2seq[u] for u in uids]
    print(f"eval proteins: {len(uids)} (val held-out: {len(val_uids & set(uids))}) | dev {dev}")

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg, meta = ck["cfg"], ck["meta"]
    model = CtrlESMC(**cfg)
    model.load_state_dict(ck["model"])
    model = model.to(dev)
    print(f"loaded {args.name} step={ck.get('step')} tokens={ck.get('tokens',0)/1e6:.0f}M | "
          f"block {args.layer}/{cfg['n_layers']} | d_model {cfg['d_model']} | "
          f"causal={cfg['causal']}")

    X, lengths = extract_layer(model, uids, seqs, args.layer, meta["aa2id"],
                               meta["bos"], meta["eos"], meta["pad"], dev)
    D = X.shape[1]
    print(f"extracted X {X.shape}")

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    is_val = np.array([u in val_uids for u in uids], dtype=bool)
    tr_rows = (np.concatenate([np.arange(offsets[i], offsets[i] + lengths[i])
                               for i in range(len(uids)) if not is_val[i]])
               if (~is_val).any() else np.arange(X.shape[0]))

    norm_scale = compute_norm_scale(X[tr_rows])
    print(f"norm_scale {norm_scale:.6f}")
    sae = train_sae((X[tr_rows] * norm_scale).astype(np.float32), input_dim=D,
                    device=sdev, epochs=args.sae_epochs,
                    expansion=args.expansion, k_sparse=args.k_sparse, k_aux=64)

    out = Path(args.out_root) / args.name / f"layer_{args.layer}"
    out.mkdir(parents=True, exist_ok=True)
    Z, _ = extract_sae_features(sae, (X * norm_scale).astype(np.float32),
                                device=sdev, save_dir=str(out))
    seqs_trunc = [s[:int(lengths[i])] for i, s in enumerate(seqs)]
    np.save(out / "Z.npy", Z.astype(np.float16))
    np.save(out / "lengths.npy", lengths.astype(np.int32))
    np.save(out / "offsets.npy", offsets)
    (out / "sequences.json").write_text(json.dumps(seqs_trunc))
    (out / "uids.json").write_text(json.dumps(uids))
    torch.save(sae.state_dict(), out / "sae_model.pt")
    (out / "META.json").write_text(json.dumps({
        "model": args.name, "layer": args.layer, "embed_dim": D,
        "sae_hidden_dim": D * args.expansion, "k_sparse": args.k_sparse,
        "norm_scale": norm_scale, "val_uids": sorted(val_uids & set(uids)),
        "ckpt": args.ckpt, "ckpt_tokens": int(ck.get("tokens", 0)),
        "causal": cfg["causal"], "anchor": "ESM-C (esm==3.2.3 TransformerStack)",
    }, indent=2))
    print(f"Z {Z.shape} sparsity {(Z==0).mean()*100:.1f}%  ->  {out}")


if __name__ == "__main__":
    main()
