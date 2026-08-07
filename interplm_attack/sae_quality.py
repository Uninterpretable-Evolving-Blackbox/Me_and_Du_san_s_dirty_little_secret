#!/usr/bin/env python3
"""
sae_quality.py — outcome-blind SAE quality metrics.

Choosing SAE capacity by the metric you are about to report is circular. These
are the criteria that are blind to concept-F1 and are the only ones allowed to
select a configuration:

    var_expl   1 - SSE/Var(x)   reconstruction fidelity
    dead       features never active on the eval set
    L0         mean active features per residue  <-- the one that matters most
    max_frac   largest share of total activation mass on a single feature
               (degeneracy tell: a collapsed dictionary can still reconstruct)

WHY L0 IS THE ONE THAT MATTERS
------------------------------
On the 33.2M pair, low L1 gave var_expl 0.998 and few dead features -- which
looks healthy until you see L0 of ~1300 active features out of 1920. That is not
a sparse autoencoder; it reconstructs by being nearly dense. This project's own
rule rejects dictionaries whose reconstruction is above a quality threshold for
exactly that reason. Always read var_expl and L0 together.

USAGE
  python sae_quality.py --sae <dir>/ae.pt --embd <analysis_embd_dir> --tag mlm_l1_0.06 --header
"""
import argparse
import glob
import os
import sys

import torch


def _import_relusae(repo):
    sys.path.insert(0, repo)
    from interplm.sae.dictionary import ReLUSAE
    return ReLUSAE


def load_acts(embd_dir, max_rows):
    chunks, total = [], 0
    files = sorted(glob.glob(os.path.join(embd_dir, "shard_*", "embeddings.pt"))) or \
            sorted(glob.glob(os.path.join(embd_dir, "shard_*", "activations.pt")))
    if not files:
        raise SystemExit(f"  no shard_*/embeddings.pt or activations.pt under {embd_dir}")
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        X = d["embeddings"] if isinstance(d, dict) else d
        chunks.append(X)
        total += X.shape[0]
        if total >= max_rows:
            break
    return torch.cat(chunks, dim=0)[:max_rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--embd", required=True)
    ap.add_argument("--repo", default=os.path.expanduser(
        "~/own_sae_data/interplm_stress/interplm_repo"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--header", action="store_true")
    args = ap.parse_args()

    ReLUSAE = _import_relusae(args.repo)
    sae = ReLUSAE.from_pretrained(args.sae, device="cpu")
    sae.eval()
    X = load_acts(args.embd, args.max_rows)
    n = X.shape[0]

    sse, l0_sum, live, mass = 0.0, 0.0, None, None
    with torch.no_grad():
        for i in range(0, n, args.batch):
            xb = X[i:i + args.batch].float()
            f = sae.encode(xb)
            xh = sae.decode(f)
            sse += torch.sum((xb - xh) ** 2).item()
            act = f > 0
            live = act.any(0) if live is None else (live | act.any(0))
            l0_sum += act.sum().item()
            m = f.sum(0)
            mass = m if mass is None else mass + m

    var = torch.var(X.float(), dim=0, unbiased=False).sum().item() * n
    ve = 1.0 - sse / var if var > 0 else float("nan")
    dead = int((~live).sum().item())

    if args.header:
        print("%-24s %9s %7s %7s %8s %9s"
              % ("tag", "var_expl", "dead", "live", "L0", "max_frac"))
    print("%-24s %9.4f %7d %7d %8.1f %9.4f"
          % (args.tag or os.path.basename(os.path.dirname(args.sae)),
             ve, dead, live.numel() - dead, l0_sum / n,
             (mass.max() / mass.sum()).item() if mass.sum() > 0 else float("nan")))


if __name__ == "__main__":
    main()
