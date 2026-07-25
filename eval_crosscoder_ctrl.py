#!/usr/bin/env python3
"""
eval_crosscoder_ctrl.py — the MLM-vs-CLM model-diffing crosscoder (the headline pair).

WHAT THIS ASKS, that the existing SAE comparison cannot
  Every result so far trains a SEPARATE SAE per arm and compares summary statistics of their
  features. The two dictionaries have no feature correspondence, so "this feature exists under
  one objective and not the other" is not even expressible. A crosscoder gives ONE shared
  dictionary with per-arm decoders, so it is:

      Are the structurally-local latents SHARED between the masked and causal models, or
      UNIQUE to the masked one?

WHY THIS PAIR IS THE HEADLINE (and ESM-2/RITA is only the extension)
  Both arms here are the SAME architecture — one ESM-C-anchored backbone, d=320, same init,
  same data order, only the objective differs. That matters concretely: the field's validity
  diagnostic for crosscoder diffing, Latent Scaling (Minder et al. 2025 Eq.4), dots the assigned
  arm's decoder direction into the OTHER arm's activation space, which REQUIRES equal widths.
  It is therefore fully applicable here and NOT applicable to ESM-2 (1280) vs RITA (1536).
  This pair is the only one where the shared/unique claim can be confirmed, not just prevented.

PIPELINE (identical to the Mac-side ESM-2/RITA run — the consistency principle)
  extract both arms -> per-token norm -> crosscoder -> validity gate -> Delta_norm
  -> Latent Scaling -> L_struct on the shared latents -> cross-tab L_struct x Delta_norm

  Run in three CLI steps, matching this repo's existing style:
    1. python eval_crosscoder_ctrl.py --ckpt-a A --ckpt-b B --layer L      (train + extract)
    2. python cpu_stage.py --layer-dir <out> --model-type residue --n-shuffles 5
    3. python eval_crosscoder_ctrl.py --crosstab <out>                     (merge + summarise)
  `run_crosscoder_ctrl.sh` does all three for you.

PREREQ: the trained checkpoints under ~/own_sae_data (do NOT delete them) + cache/pdb_files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_ctrl_esmc import CtrlESMC
from eval_ctrl_plm import pick_device, load_eval_set, extract_layer
from crosscoder import (train_crosscoder, encode_all, val_ev, delta_norm,
                        latent_scaling, dead_stats)

DEGENERATE_EV = 0.99


def per_token_norm(X):
    """RMS-normalise each residue to ||x|| = sqrt(D) (Templeton 2024) — the C2/A2 instrument fix.

    Also equalises the two arms' scales so whichever arm has larger activations cannot dominate
    the joint reconstruction loss.
    """
    D = X.shape[1]
    for i in range(0, X.shape[0], 20000):
        xb = X[i:i + 20000]
        nrm = np.linalg.norm(xb, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X[i:i + 20000] = (xb / nrm * np.sqrt(D)).astype(X.dtype)
    return X


def load_arm(ckpt, uids, seqs, layer, device):
    ck = torch.load(ckpt, map_location="cpu")
    cfg, meta = ck["cfg"], ck["meta"]
    model = CtrlESMC(**cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device)
    X, lengths = extract_layer(model, uids, seqs, layer, meta["aa2id"],
                               meta["bos"], meta["eos"], meta["pad"], device)
    print(f"  {Path(ckpt).parent.name}: block {layer}/{cfg['n_layers']} causal={cfg['causal']} "
          f"tok={ck.get('tokens',0)/1e9:.2f}B -> X{X.shape}")
    del model
    return X, lengths, cfg


def crosstab(out_dir):
    """Step 3: merge L_struct onto the latent table and write the summary."""
    out = Path(out_dir)
    lat = pd.read_csv(out / "latent_diffing.csv")
    sd = pd.read_csv(out / "struct_seq_metrics.csv")
    lat = lat.merge(sd[["feature_idx", "struct_delta"]].rename(columns={"feature_idx": "latent"}),
                    on="latent", how="left")
    lat.to_csv(out / "latent_diffing.csv", index=False)

    meta = json.loads((out / "META.json").read_text())
    a, b = meta["arm_a"], meta["arm_b"]

    # PRIMARY statistic: Spearman rho — invariant to monotone rescaling of Delta_norm, so it
    # survives the global shift described below.
    rho = lat[["delta_norm", "struct_delta"]].corr(method="spearman").iloc[0, 1]

    # QUANTILE bins are primary. Under BatchTopK the Delta_norm distribution is deliberately far
    # less spread than under L1 (killing that spurious trimodality is the point of Minder et al.
    # 2025), and it shifts globally toward whichever arm reconstructs more easily. Absolute cut
    # points would drop every latent into one "shared" bin and report nothing.
    q = np.unique(lat["delta_norm"].quantile([0, .1, .3, .7, .9, 1.0]).values)
    qlabels = [f"{a}-most", f"{a}-lean", "middle", f"{b}-lean", f"{b}-most"][:len(q) - 1]
    lat["dn_qbin"] = pd.cut(lat["delta_norm"], bins=q, labels=qlabels, include_lowest=True)
    qtab = lat.groupby("dn_qbin", observed=False)["struct_delta"].agg(["count", "mean", "median"])
    print(f"\n  L_struct by Delta_norm DECILE ({a}-leaning ... {b}-leaning):")
    print(qtab.to_string())

    lat["dn_bin"] = pd.cut(lat["delta_norm"], bins=[0, .1, .3, .7, .9, 1.0],
                           labels=[f"{a}-only", f"{a}-lean", "shared", f"{b}-lean", f"{b}-only"],
                           include_lowest=True)
    atab = lat.groupby("dn_bin", observed=False)["struct_delta"].agg(["count", "mean"])
    print(f"  Delta_norm median {lat['delta_norm'].median():.3f} "
          f"[10-90: {lat['delta_norm'].quantile(.1):.3f}, {lat['delta_norm'].quantile(.9):.3f}]"
          f" | absolute bins occupied {int((atab['count'] > 0).sum())}/5")
    print(f"  Spearman(Delta_norm, struct_delta) = {rho:+.4f}   <-- PRIMARY")

    def _tab(t):
        return {str(i): {k: (None if pd.isna(v) else float(v)) for k, v in r.items()}
                for i, r in t.iterrows()}
    meta["crosstab_quantile"] = _tab(qtab)
    meta["crosstab_absolute"] = _tab(atab)
    meta["delta_norm_median"] = float(lat["delta_norm"].median())
    meta["delta_norm_p10_p90"] = [float(lat["delta_norm"].quantile(.1)),
                                  float(lat["delta_norm"].quantile(.9))]
    meta["spearman_dn_lstruct"] = float(rho)
    (out / "META.json").write_text(json.dumps(meta, indent=2))
    lat.to_csv(out / "latent_diffing.csv", index=False)
    print(f"  -> {out}/META.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosstab", default=None, help="step 3: merge L_struct into an existing dir")
    ap.add_argument("--ckpt-a", help="reference arm (sets hidden/k); e.g. the MLM checkpoint")
    ap.add_argument("--ckpt-b", help="comparison arm; e.g. the CLM checkpoint")
    ap.add_argument("--name-a", default="mlm")
    ap.add_argument("--name-b", default="clm")
    ap.add_argument("--layer", type=int, help="block index, same in both arms")
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--out-root", default="outputs_crosscoder")
    ap.add_argument("--seed", type=int, default=42, help="crosscoder init; paper uses 42/43/44")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k-frac", type=float, default=0.20, help="k/d_A; 0.20 == the paper's 256/1280")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-proteins", type=int, default=0)
    args = ap.parse_args()

    if args.crosstab:
        crosstab(args.crosstab)
        return
    for req in ("ckpt_a", "ckpt_b", "layer"):
        if getattr(args, req) in (None,):
            raise SystemExit(f"--{req.replace('_','-')} is required")

    dev = pick_device(args.device)
    uids, uid2seq, val_uids = load_eval_set(args.eval_set)
    uids = [u for u in uids if u in uid2seq]
    if args.max_proteins:
        uids = uids[:args.max_proteins]
    seqs = [uid2seq[u] for u in uids]
    print(f"eval proteins {len(uids)} (val {len(val_uids & set(uids))}) | layer {args.layer} | dev {dev}")

    XA, lenA, cfgA = load_arm(args.ckpt_a, uids, seqs, args.layer, dev)
    XB, lenB, cfgB = load_arm(args.ckpt_b, uids, seqs, args.layer, dev)

    # ---- alignment gate: row i must be the same residue in both arms ----
    if not np.array_equal(lenA, lenB):
        raise SystemExit("lengths differ between arms — rows are not comparable")
    if XA.shape[0] != XB.shape[0]:
        raise SystemExit(f"row mismatch {XA.shape[0]} vs {XB.shape[0]}")
    lengths = lenA

    per_token_norm(XA); per_token_norm(XB)
    print(f"  per-token RMS-normalised both arms to sqrt(D)={np.sqrt(XA.shape[1]):.1f}")

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    is_val = np.array([u in val_uids for u in uids], dtype=bool)
    tr = np.ones(XA.shape[0], dtype=bool)
    for i, v in enumerate(is_val):
        if v:
            tr[offsets[i]:offsets[i] + lengths[i]] = False

    hidden = args.expansion * XA.shape[1]
    k = max(1, int(round(args.k_frac * XA.shape[1])))
    print(f"  crosscoder: hidden {hidden} | k {k} | k/embed {k/XA.shape[1]*100:.1f}% "
          f"| k/hidden {k/hidden*100:.2f}% | seed {args.seed}")

    cc = train_crosscoder(XA[tr], XB[tr], hidden_dim=hidden, k_sparse=k, device=dev,
                          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                          seed=args.seed)

    ev = val_ev(cc, XA[~tr], XB[~tr], device=dev)
    Z = encode_all(cc, XA, XB, device=dev)
    ds = dead_stats(cc, Z)
    degenerate = (ev["val_EV_a"] >= DEGENERATE_EV) or (ev["val_EV_b"] >= DEGENERATE_EV)
    print(f"  val_EV {args.name_a} {ev['val_EV_a']:.4f}  {args.name_b} {ev['val_EV_b']:.4f}"
          f"{'   <-- DEGENERATE' if degenerate else ''}")
    print(f"  dead {ds['pct_dead']:.1f}%  always-on {ds['pct_always_on_90']:.1f}%  "
          f"mean L0 {ds['mean_l0']:.1f}")

    dn = delta_norm(cc)
    assign = (dn["delta_norm"] >= 0.5).astype(int)
    ls = latent_scaling(cc, XA, XB, assign, device=dev)
    if ls["applicable"]:
        print(f"  latent scaling: median nu {np.nanmedian(ls['nu']):.3f} | "
              f"nu_eps {np.nanmedian(ls['nu_eps']):.3f} (Complete Shrinkage) | "
              f"nu_r {np.nanmedian(ls['nu_r']):.3f} (Latent Decoupling)")
        print("     nu near 0 => genuinely arm-specific; near 1 => the 'unique' label is spurious")
    else:
        print(f"  !! Latent Scaling NOT APPLICABLE: {ls['reason']}")

    out = Path(args.out_root) / f"cc_{args.name_a}_{args.name_b}_l{args.layer}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Z.npy", Z.astype(np.float16))
    np.save(out / "lengths.npy", lengths.astype(np.int32))
    np.save(out / "offsets.npy", offsets)
    (out / "uids.json").write_text(json.dumps(uids))
    (out / "sequences.json").write_text(json.dumps([s[:int(lengths[i])] for i, s in enumerate(seqs)]))
    torch.save(cc.state_dict(), out / "crosscoder.pt")

    lat = pd.DataFrame({"latent": np.arange(hidden), **dn})
    if ls["applicable"]:
        lat["nu"], lat["nu_eps"], lat["nu_r"] = ls["nu"], ls["nu_eps"], ls["nu_r"]
    lat.to_csv(out / "latent_diffing.csv", index=False)

    (out / "META.json").write_text(json.dumps({
        "arm_a": args.name_a, "arm_b": args.name_b, "layer": args.layer,
        "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
        "embed_dim": int(XA.shape[1]), "sae_hidden_dim": hidden, "k_sparse": k,
        "sae_seed": args.seed, "per_token_norm": True, "epochs": args.epochs,
        "val_uids": sorted(val_uids & set(uids)),
        "anchor": "ESM-C (esm==3.2.3 TransformerStack)",
        "causal_a": cfgA["causal"], "causal_b": cfgB["causal"],
        **ev, "degenerate": bool(degenerate), **ds,
        "latent_scaling_applicable": bool(ls["applicable"]),
        "latent_scaling_note": ls.get("reason", ""),
    }, indent=2))
    print(f"  -> {out}   (next: cpu_stage.py --layer-dir {out} --model-type residue --n-shuffles 5)")


if __name__ == "__main__":
    main()
