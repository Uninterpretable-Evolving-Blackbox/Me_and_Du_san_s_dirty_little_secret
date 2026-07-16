#!/usr/bin/env python3
"""
measure_rank_ev.py — the SAE-VALIDITY diagnostic for the controlled experiment.

Why this exists: L_struct is computed on SAE features. An SAE is only a valid
instrument when sparsity forces a real bottleneck. If MLM's and CLM's activations
occupy very different numbers of effective dimensions, the SAE lands in a different
regime on each arm, and any L_struct difference may be an instrument artifact rather
than an objective effect (this is exactly why ProGen2 was dropped at val_EV 0.99).

This measures, per layer, per arm, straight from a trained checkpoint:

  1. EFFECTIVE RANK of the raw per-residue activations (NO SAE involved):
       - participation ratio  PR   = (sum lambda)^2 / sum(lambda^2)
       - entropy eff. rank    eRank = exp(-sum p_i ln p_i),  p_i = lambda_i / sum lambda
     (lambda = eigenvalues of the DxD activation covariance; D = d_model.)
     This is the SAE-FREE finding: how many dimensions the model actually uses.

  2. SAE val_EV (needs an SAE): trains the same TopK SAE eval_ctrl_plm.py uses and
     reports val explained variance. >= 0.99 == degenerate basis (kill threshold).

Usage:
  python measure_rank_ev.py --ckpt CKPT --name mlm_token --eval-set eval_set \
      --out results_rank_ev/summary.json                 # rank + EV, all 9 depths
  python measure_rank_ev.py ... --no-ev                  # rank only (fast, no SAE)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_ctrl_esmc import CtrlESMC
from eval_ctrl_plm import (pick_device, load_eval_set, extract_layer,
                           effective_rank, sae_val_ev)
from train_sae import compute_norm_scale, train_sae


# nine paper-matched relative depths over 30 blocks (index/29 ~ paper's index/32)
DEFAULT_LAYERS = [0, 4, 7, 11, 14, 18, 22, 26, 29]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="e.g. mlm_token / clm / mlm_pred")
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--out", default="results_rank_ev/summary.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--no-ev", dest="with_ev", action="store_false", default=True,
                    help="skip the SAE val_EV (rank only; no SAE trained)")
    ap.add_argument("--sae-epochs", type=int, default=60)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k-sparse", type=int, default=256)
    ap.add_argument("--sae-seed", type=int, default=42)
    args = ap.parse_args()

    dev = pick_device(args.device)
    uids, uid2seq, val_uids = load_eval_set(args.eval_set)
    uids = [u for u in uids if u in uid2seq]
    if args.max_proteins:
        uids = uids[:args.max_proteins]
    seqs = [uid2seq[u] for u in uids]

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg, meta = ck["cfg"], ck["meta"]
    model = CtrlESMC(**cfg)
    model.load_state_dict(ck["model"])
    model = model.to(dev)
    print(f"{args.name}: {cfg['d_model']}d/{cfg['n_layers']}L causal={cfg['causal']} | "
          f"{len(uids)} proteins | dev {dev} | EV={'on' if args.with_ev else 'off'}")

    layers = [int(x) for x in args.layers.split(",")]
    rows = []
    for L in layers:
        X, lengths = extract_layer(model, uids, seqs, L, meta["aa2id"],
                                   meta["bos"], meta["eos"], meta["pad"], dev)
        rk = effective_rank(X)
        rec = dict(name=args.name, layer=L, n_residues=int(X.shape[0]),
                   d_model=int(X.shape[1]), **rk)
        if args.with_ev:
            offs = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
            is_val = np.array([u in val_uids for u in uids], dtype=bool)
            rowmask = np.zeros(X.shape[0], dtype=bool)
            for i, v in enumerate(is_val):
                if v:
                    rowmask[offs[i]:offs[i] + lengths[i]] = True
            ns = compute_norm_scale(X[~rowmask])
            torch.manual_seed(args.sae_seed); np.random.seed(args.sae_seed)
            sae = train_sae((X[~rowmask] * ns).astype(np.float32), input_dim=X.shape[1],
                            device=dev, epochs=args.sae_epochs,
                            expansion=args.expansion, k_sparse=args.k_sparse, k_aux=64)
            rec["val_EV"] = float(sae_val_ev(sae, X[rowmask] * ns, dev))
            rec["degenerate"] = bool(rec["val_EV"] >= 0.99)
        rows.append(rec)
        ev = f" val_EV {rec['val_EV']:.4f}{'  <-- DEGENERATE' if rec.get('degenerate') else ''}" if args.with_ev else ""
        print(f"  L{L:>2}: PR {rk['participation_ratio']:6.1f}  eRank {rk['entropy_erank']:6.1f}  "
              f"/ {rec['d_model']}d{ev}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text())["rows"] if out.exists() else []
    prev = [r for r in prev if r["name"] != args.name]
    out.write_text(json.dumps(dict(
        note="Effective rank (SAE-free) + SAE val_EV per depth. PR=participation ratio, "
             "eRank=entropy effective rank. val_EV>=0.99 => degenerate SAE basis (kill).",
        rows=prev + rows), indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
