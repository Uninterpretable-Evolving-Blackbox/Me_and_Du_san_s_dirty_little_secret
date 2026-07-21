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


def effective_rank(X: np.ndarray):
    """Effective rank of activations X (N, D), SAE-free, WITH outlier control.

    Raw participation ratio is dominated by "massive activation" outlier dimensions:
    a single rogue dim can drive PR to ~1 even when the rest of the representation is
    high-rank (real RITA does exactly this - one dim holds ~97% of variance). So we
    report PR after dropping the top-1 and top-5 eigenvalues, plus the top-1 share
    itself, which is the massive-activation diagnostic.

      participation_ratio : raw PR  (comparable across models ONLY with top1_share)
      pr_drop1 / pr_drop5 : PR of the spectrum with the top 1 / 5 components removed
      top1_share          : fraction of total variance in the largest component
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    lam = np.clip(np.linalg.eigvalsh(cov.astype(np.float64)), 0, None)
    lam = np.sort(lam)[::-1]                       # descending
    s = float(lam.sum())
    if s <= 0:
        return dict(participation_ratio=0.0, entropy_erank=0.0,
                    pr_drop1=0.0, pr_drop5=0.0, top1_share=0.0, dims_90pct=0)

    def _pr(l):
        t = float(l.sum())
        return float((t * t) / float(np.sum(l * l))) if t > 0 else 0.0

    p = lam / s
    p = p[p > 0]
    return dict(participation_ratio=_pr(lam),
                entropy_erank=float(np.exp(-np.sum(p * np.log(p)))),
                pr_drop1=_pr(lam[1:]),
                pr_drop5=_pr(lam[5:]),
                top1_share=float(lam[0] / s),
                dims_90pct=int(np.searchsorted(np.cumsum(lam) / s, 0.90)) + 1)


@torch.no_grad()
def sae_val_ev(sae, X_val_scaled, sdev):
    """SAE val explained variance on already-norm-scaled held-out residues."""
    xb = torch.from_numpy(X_val_scaled.astype(np.float32)).to(sdev)
    z, _ = sae.encode(xb)
    recon = sae.decode(z)
    num = ((xb - recon) ** 2).sum().item()
    den = ((xb - xb.mean(0)) ** 2).sum().item()
    return 1.0 - num / den if den > 0 else float("nan")


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
    ap.add_argument("--expansion", type=int, default=8)     # paper: 8x
    ap.add_argument("--k-sparse", type=int, default=256)    # paper: k=256
    ap.add_argument("--sae-seed", type=int, default=42,
                    help="SAE init seed; paper's main grid is 42 (43/44 = robustness)")
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
    # Seed per (model, layer, sae_seed) immediately before training, exactly as
    # run_unsupervised.py:447 does, so the SAE init is deterministic instead of
    # drifting with whatever RNG state preceded it. Paper default: 42.
    torch.manual_seed(args.sae_seed)
    np.random.seed(args.sae_seed)
    print(f"SAE seed {args.sae_seed} (expansion={args.expansion}, k={args.k_sparse}, k_aux=64)")
    sae = train_sae((X[tr_rows] * norm_scale).astype(np.float32), input_dim=D,
                    device=sdev, epochs=args.sae_epochs,
                    expansion=args.expansion, k_sparse=args.k_sparse, k_aux=64)

    out = Path(args.out_root) / args.name / f"layer_{args.layer}"
    out.mkdir(parents=True, exist_ok=True)
    Z, _ = extract_sae_features(sae, (X * norm_scale).astype(np.float32),
                                device=sdev, save_dir=str(out))

    # SAE-VALIDITY diagnostic (the check whose absence invalidated the earlier pilot):
    # val_EV >= 0.99 == degenerate basis; effective rank tells whether the two arms'
    # SAEs are even in comparable regimes. Computed here where X is still in memory.
    val_rows = np.zeros(X.shape[0], dtype=bool)
    for i in range(len(uids)):
        if is_val[i]:
            val_rows[offsets[i]:offsets[i] + lengths[i]] = True
    ev = sae_val_ev(sae, X[val_rows] * norm_scale, sdev) if val_rows.any() else float("nan")
    rank = effective_rank(X)
    print(f"val_EV {ev:.4f}{'  <-- DEGENERATE (>=0.99)' if ev >= 0.99 else ''} | "
          f"eff.rank PR {rank['participation_ratio']:.1f} eRank {rank['entropy_erank']:.1f} / {D}d")

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
        "sae_seed": args.sae_seed,
        "val_EV": float(ev), "degenerate": bool(ev >= 0.99),
        "participation_ratio": rank["participation_ratio"],
        "entropy_erank": rank["entropy_erank"],
    }, indent=2))
    print(f"Z {Z.shape} sparsity {(Z==0).mean()*100:.1f}%  ->  {out}")


if __name__ == "__main__":
    main()
