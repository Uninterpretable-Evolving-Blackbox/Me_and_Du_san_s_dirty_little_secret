#!/usr/bin/env python3
"""
experiment_raw_coactivation.py — L_struct with no autoencoder, plus perplexity.

WHY THIS EXISTS
---------------
Two reviewers independently asked for the same thing.

  * L_struct has only ever been computed on SAE features. So the metric has never
    been compared against its own object: the model's raw activations. If the
    shuffled/real gap persists on raw dimensions, no instrumental account of the
    failure can hold, because there is no instrument. If it collapses, the
    autoencoder is the story. Section 4.3 currently cannot choose between those --
    its estimate of the instrumental share ranges from 4% to 89% depending on which
    dose-response is fitted -- and this run settles it directly.

  * Nobody has ever recorded perplexity for these models. Without it there is no
    evidence the models trained properly at all, which is the first thing a reader
    asks before believing any negative result about them. It is nearly free once
    the checkpoint is loaded, so it is computed here rather than in its own stage.

WHAT IT DOES
------------
Runs the IDENTICAL estimator used for every L_struct number in this project
(_process_struct_seq_chunk_v3 from cpu_stage) but hands it the raw per-residue
activation matrix (n_res x d_model) instead of the SAE code matrix (n_res x n_feat).
Same graphs, same permutation null, same top-k gate, same struct_delta = obs - shuffled.

Perplexity is exp(mean NLL per residue) on the held-out proteins. For the causal arm
this is a genuine held-out likelihood. For the masked arm it is a PSEUDO-perplexity
(one position masked at a time), which is a different quantity and is NOT comparable
across arms -- the output labels it accordingly so the two are never put in a table
together.

USAGE
  python experiment_raw_coactivation.py --ckpt <model_final.pt> --name ckpt_clm_s42 \
      --layers 11,14,18 --out results_raw_coactivation/ckpt_clm_s42.csv

  # metric only, from a saved raw matrix (no checkpoint needed):
  python experiment_raw_coactivation.py --raw-dir <dir with X.npy> --skip-perplexity
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from cpu_stage import (
    load_ref_seqs, build_neighbor_graphs_residue_parallel,
    adj_list_to_sparse, build_protein_permutations,
    _process_struct_seq_chunk_v3,
)


def struct_delta_on_matrix(X, uids, lengths, ref_seqs, pdb_dir,
                           cutoff, seq_gap, n_shuffles, n_jobs, topk_frac):
    """The project's standard estimator, applied to an arbitrary (n_res, d) matrix."""
    n_res = int(np.sum(lengths))
    if X.shape[0] != n_res:
        raise SystemExit(f"row mismatch: X has {X.shape[0]} rows, lengths sum to {n_res}")

    seq_adj, struct_adj = build_neighbor_graphs_residue_parallel(
        uids, lengths, ref_seqs, Path(pdb_dir), n_jobs,
        contact_cutoff=cutoff, seq_gap_min=seq_gap)
    A_seq, deg_seq = adj_list_to_sparse(seq_adj, n_res)
    A_struct, deg_struct = adj_list_to_sparse(struct_adj, n_res)
    del seq_adj, struct_adj

    perms = build_protein_permutations(lengths, n_shuffles)
    n_feat = int(X.shape[1])
    chunk = 64
    n_chunks = (n_feat + chunk - 1) // chunk
    res = Parallel(n_jobs=n_jobs)(
        delayed(_process_struct_seq_chunk_v3)(
            ci, chunk, X, None, A_seq, deg_seq, A_struct, deg_struct,
            perms, n_feat, topk_frac)
        for ci in range(n_chunks))
    idx = np.concatenate([r[0] for r in res])
    seq_obs = np.concatenate([r[1] for r in res])
    str_obs = np.concatenate([r[2] for r in res])
    seq_sh = np.concatenate([r[3] for r in res])
    str_sh = np.concatenate([r[4] for r in res])
    o = np.argsort(idx)
    return dict(struct_delta=(str_obs - str_sh)[o], seq_delta=(seq_obs - seq_sh)[o],
                n_edges=int(A_struct.nnz))


@torch.no_grad()
def perplexity(model, uids, seqs, aa2id, bos, eos, pad, device, causal, max_prot=150):
    """exp(mean NLL per residue) on held-out proteins.

    causal=True  -> true left-to-right perplexity.
    causal=False -> PSEUDO-perplexity, one masked position at a time. Different
                    quantity; never compare the two across arms.
    """
    mask_id = aa2id.get("<mask>", aa2id.get("X"))
    tot_nll, tot_n = 0.0, 0
    for uid in uids[:max_prot]:
        ids = [bos] + [aa2id.get(c, aa2id.get("X")) for c in seqs[str(uid)]] + [eos]
        x = torch.tensor([ids], device=device)
        if causal:
            logits = model(x)["logits"] if isinstance(model(x), dict) else model(x)
            lp = torch.log_softmax(logits[0, :-1].float(), -1)
            tgt = x[0, 1:]
            tot_nll += float(-lp[torch.arange(len(tgt)), tgt].sum()); tot_n += len(tgt)
        else:
            L = x.shape[1]
            for i in range(1, L - 1):                      # skip BOS/EOS
                xm = x.clone(); xm[0, i] = mask_id
                out = model(xm)
                logits = out["logits"] if isinstance(out, dict) else out
                lp = torch.log_softmax(logits[0, i].float(), -1)
                tot_nll += float(-lp[x[0, i]]); tot_n += 1
    return math.exp(tot_nll / max(tot_n, 1)), tot_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--raw-dir", default=None,
                    help="dir containing X.npy + uids.json + lengths.npy + sequences.json")
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--layers", default="11,14,18")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--seq-gap", type=int, default=12)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--topk-frac", type=float, default=0.10)
    ap.add_argument("--skip-perplexity", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.raw_dir:
        d = Path(args.raw_dir)
        X = np.load(d / "X.npy").astype(np.float32)
        uids = [str(u) for u in json.loads((d / "uids.json").read_text())]
        lengths = np.load(d / "lengths.npy")
        ref_seqs = load_ref_seqs(d)
        r = struct_delta_on_matrix(X, uids, lengths, ref_seqs, args.pdb_dir,
                                   args.cutoff, args.seq_gap, args.n_shuffles,
                                   args.n_jobs, args.topk_frac)
        rows.append(dict(source=str(d), layer=-1, d_model=X.shape[1],
                         mean_struct_delta=float(r["struct_delta"].mean()),
                         mean_seq_delta=float(r["seq_delta"].mean()),
                         n_edges=r["n_edges"], perplexity=None, ppl_kind=None))
    else:
        if not (args.ckpt and args.name):
            raise SystemExit("need --ckpt and --name (or --raw-dir)")
        from model_ctrl_esmc import CtrlESMC
        from eval_ctrl_plm import load_eval_set, extract_layer, pick_device
        device = pick_device(args.device)
        ck = torch.load(args.ckpt, map_location=device)
        model = CtrlESMC(**ck["config"]) if "config" in ck else CtrlESMC()
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        model.to(device).eval()
        causal = bool(ck.get("config", {}).get("causal", "clm" in args.name))

        uids, seqs, aa2id, bos, eos, pad = load_eval_set(args.eval_set)
        uids = [str(u) for u in uids]
        ref_seqs = {u: seqs[u] for u in uids}
        lengths = np.array([len(seqs[u]) for u in uids], dtype=np.int32)

        ppl, ppl_n = (None, 0)
        if not args.skip_perplexity:
            ppl, ppl_n = perplexity(model, uids, seqs, aa2id, bos, eos, pad,
                                    device, causal)
            print(f"  perplexity ({'causal' if causal else 'pseudo, masked'}): "
                  f"{ppl:.3f} over {ppl_n:,} residues")

        for L in [int(x) for x in args.layers.split(",")]:
            X = extract_layer(model, uids, seqs, L, aa2id, bos, eos, pad, device)
            X = np.asarray(X, dtype=np.float32)
            r = struct_delta_on_matrix(X, uids, lengths, ref_seqs, args.pdb_dir,
                                       args.cutoff, args.seq_gap, args.n_shuffles,
                                       args.n_jobs, args.topk_frac)
            print(f"  L{L}: mean raw struct_delta {r['struct_delta'].mean():+.5f} "
                  f"(d_model={X.shape[1]}, {r['n_edges']:,} edges)")
            rows.append(dict(source=args.name, layer=L, d_model=int(X.shape[1]),
                             mean_struct_delta=float(r["struct_delta"].mean()),
                             mean_seq_delta=float(r["seq_delta"].mean()),
                             n_edges=r["n_edges"], perplexity=ppl,
                             ppl_kind="causal" if causal else "pseudo_masked"))

    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  written: {out}")
    print("  compare mean_struct_delta between the real and shuffled trees.")
    print("  if the gap persists here, no dictionary-based account of it can hold.")


if __name__ == "__main__":
    main()
