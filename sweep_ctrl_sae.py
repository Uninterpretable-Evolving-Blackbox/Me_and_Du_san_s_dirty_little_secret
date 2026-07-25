#!/usr/bin/env python3
"""
sweep_ctrl_sae.py — the k x expansion sweep for the CONTROLLED models (rigor item A8).

WHY THIS EXISTS
  The project's SAE hyperparameters (k=256, expansion 8x) were ablated on ESM-2 and RITA:
  6 layers x 12 configs = 72 runs. Zero of those runs are on the controlled models, and the
  transfer has ALREADY FAILED TWICE:

    * k=256 was chosen at d=1280 (k/embed = 20%). Applied to the d=480 controlled hybrid it
      became k/embed = 53% — barely a sparse decomposition, val_EV 0.945-0.997. Appendix O had
      to be cut.
    * Re-anchoring to k=96 by matching BOTH ratios (20% / 2.5%) still failed, because the real
      problem was the MLM-vs-CLM effective-rank gap (38 vs 282 dims at 25% depth). The project's
      own conclusion: "no single k fixes it — the rank gap IS the effect."

  So the controlled pair needs its own sweep, and — critically — the sweep must be read with the
  rank gap in mind: a config that is well-conditioned for one arm may be degenerate for the
  other. That asymmetry is the finding, not a nuisance to tune away.

DESIGN
  Extract each arm's activations ONCE per depth, then train every (k, expansion) SAE on the
  cached array. Re-running eval_ctrl_plm.py per cell would repeat the forward pass 12x for
  nothing. Reports val_EV and the train-val EV gap per arm, plus the |gap| BETWEEN arms, which
  is the number that actually matters here: L_struct is only comparable if both arms' SAEs are
  in comparable regimes.

  Selection is outcome-blind: val_EV, EV gap and dead-count only. L_struct is never computed
  here and must never be used to pick k.

Usage
  python sweep_ctrl_sae.py --layer 14                      # both arms, 12 configs
  python sweep_ctrl_sae.py --layer 14 --ks 64,128,256 --expansions 4,8,16
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_ctrl_esmc import CtrlESMC
from eval_ctrl_plm import pick_device, load_eval_set, extract_layer, effective_rank
from train_sae import compute_norm_scale, train_sae
from sae_variants import val_ev_variant


def per_token_norm(X):
    D = X.shape[1]
    for i in range(0, X.shape[0], 20000):
        xb = X[i:i + 20000]
        n = np.linalg.norm(xb, axis=1, keepdims=True); n[n == 0] = 1.0
        X[i:i + 20000] = (xb / n * np.sqrt(D)).astype(X.dtype)
    return X


@torch.no_grad()
def _ev(sae, Xs, dev, bs=8192):
    mu = torch.from_numpy(Xs.mean(0)).to(dev)
    num = den = 0.0
    for i in range(0, Xs.shape[0], bs):
        xb = torch.from_numpy(Xs[i:i + bs]).to(dev)
        z, _ = sae.encode(xb)
        r = sae.decode(z)
        num += float(((xb - r) ** 2).sum()); den += float(((xb - mu) ** 2).sum())
    return 1.0 - num / max(den, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--arms", default="mlm_token,clm")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--ks", default="64,128,256")
    ap.add_argument("--expansions", default="4,8,16,32")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sae-seed", type=int, default=42)
    ap.add_argument("--per-token-norm", action="store_true", default=True)
    ap.add_argument("--out", default="results_ctrl_sweep/sweep.csv")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import os
    data = args.data or os.path.expanduser("~/own_sae_data/uniref50_pilot")
    ckpt_dir = {"clm": "ckpt_clm_s42", "mlm_token": "ckpt_mlm_s42_token",
                "mlm_pred": "ckpt_mlm_s42_pred"}
    dev = pick_device(args.device)
    uids, uid2seq, val_uids = load_eval_set(args.eval_set)
    uids = [u for u in uids if u in uid2seq]
    seqs = [uid2seq[u] for u in uids]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []

    ks = [int(x) for x in args.ks.split(",")]
    exps = [int(x) for x in args.expansions.split(",")]

    for arm in args.arms.split(","):
        ck = Path(data) / ckpt_dir.get(arm, f"ckpt_{arm}") / "model_final.pt"
        if not ck.exists():
            print(f"[skip] {arm}: no {ck}"); continue
        blob = torch.load(ck, map_location="cpu")
        model = CtrlESMC(**blob["cfg"]); model.load_state_dict(blob["model"]); model = model.to(dev)
        X, lengths = extract_layer(model, uids, seqs, args.layer, blob["meta"]["aa2id"],
                                   blob["meta"]["bos"], blob["meta"]["eos"],
                                   blob["meta"]["pad"], dev)
        del model
        if args.per_token_norm:
            per_token_norm(X)
        rk = effective_rank(X)
        D = X.shape[1]
        offs = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        is_val = np.array([u in val_uids for u in uids])
        tr = np.ones(X.shape[0], bool)
        for i, v in enumerate(is_val):
            if v:
                tr[offs[i]:offs[i] + lengths[i]] = False
        ns = compute_norm_scale(X[tr])
        Xs = (X * ns).astype(np.float32)
        print(f"\n=== {arm} L{args.layer} | d={D} | eff.rank PR {rk['participation_ratio']:.1f} "
              f"pr_drop5 {rk['pr_drop5']:.1f} | train {tr.sum():,} ===", flush=True)

        for e in exps:
            for k in ks:
                if any(r["arm"] == arm and r["k"] == k and r["expansion"] == e
                       and r["layer"] == args.layer for r in rows):
                    print(f"  [done] k={k} exp={e}"); continue
                torch.manual_seed(args.sae_seed); np.random.seed(args.sae_seed)
                sae = train_sae(Xs[tr], input_dim=D, device=dev, epochs=args.epochs,
                                expansion=e, k_sparse=k, k_aux=64)
                sae.eval()
                ev_v, ev_t = _ev(sae, Xs[~tr], dev), _ev(sae, Xs[tr][:40000], dev)
                fired = np.zeros(D * e, np.int64)
                with torch.no_grad():
                    for i in range(0, Xs.shape[0], 8192):
                        z, _ = sae.encode(torch.from_numpy(Xs[i:i + 8192]).to(dev))
                        fired += (z > 0).sum(0).cpu().numpy()
                dead = float((fired == 0).mean() * 100)
                rows.append(dict(arm=arm, layer=args.layer, d_model=D, k=k, expansion=e,
                                 hidden=D * e, k_over_embed=k / D, k_over_hidden=k / (D * e),
                                 val_EV=ev_v, ev_gap=ev_t - ev_v, pct_dead=dead,
                                 degenerate=bool(ev_v >= 0.99),
                                 pr_drop5=rk["pr_drop5"], top1_share=rk["top1_share"]))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"  k={k:<4} exp={e:<3} k/embed {k/D*100:5.1f}% k/hidden {k/(D*e)*100:5.2f}% "
                      f"| val_EV {ev_v:.4f} gap {ev_t-ev_v:+.4f} dead {dead:.1f}%"
                      f"{'  <-- DEGENERATE' if ev_v >= 0.99 else ''}", flush=True)

    df = pd.DataFrame(rows)
    df = df[df.layer == args.layer]
    print("\n" + "=" * 92)
    print(df.to_string(index=False))
    # the number that matters: is any config well-conditioned for BOTH arms at once?
    if df.arm.nunique() > 1:
        piv = df.pivot_table(index=["k", "expansion"], columns="arm", values="val_EV")
        piv["ev_gap_between_arms"] = (piv.max(axis=1) - piv.min(axis=1))
        piv["max_val_EV"] = piv.max(axis=1)
        print("\nBETWEEN-ARM val_EV gap (small = the two arms' SAEs are in comparable regimes):")
        print(piv.sort_values("ev_gap_between_arms").to_string())
        print("\n  A large gap at EVERY config is the rank-gap signature — it means no single "
              "\n  (k, expansion) puts both arms in the same regime, and L_struct comparisons at "
              "\n  that depth are instrument-limited rather than objective-limited. Report it.")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
