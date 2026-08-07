#!/usr/bin/env python3
"""
embed_ctrl_interplm.py — emit OUR controlled models' activations in InterPLM's
format, so their published concept-F1 runs with the backbone as the only change.

Handles BOTH model generations:
    --arch esmc   CtrlESMC  (the 42.0M pair in the paper; model_ctrl_esmc.py)
    --arch plm    PLM       (the 33.2M pair; model_ctrl_plm.py)

They differ in the tokeniser field names, which is exactly the kind of thing that
silently misaligns rows if guessed:
    esmc : meta["aa2id"], meta["bos"], meta["eos"], meta["pad"]
    plm  : meta["vocab"] (includes specials), meta["bos"], ...

OUTPUTS (matched to their own scripts)
    --shards-dir -> <out>/shard_N/embeddings.pt {embeddings, boundaries, protein_ids}
    --fasta-dir  -> <out>/shard_N/activations.pt + metadata.yaml
    metadata.yaml is REQUIRED: their dataloader silently rejects shards without it.

ROW ALIGNMENT
    extract_annotations walks shard_N/protein_data.tsv in row order, one row per
    residue of the FULL sequence, and never truncates. We walk the same file in
    the same order and ASSERT equal row counts per shard, refusing to write on
    mismatch. Any protein longer than max_seq-2 will trip this -- filter the
    protein set by length before extracting annotations.

    42.0M ESM-C models: max_seq is in cfg. 33.2M: 512.

STATUS: the --arch plm path is tested end to end on the 33.2M pair.
        The --arch esmc path is NOT tested -- no 42M checkpoint or esm==3.2.3
        was available where this was written. Run the smoke check first:
            python embed_ctrl_interplm.py --ckpt <42M ckpt> --arch esmc \\
                --smoke --layer 14
        It loads the model, embeds 8 proteins and prints shapes. If that passes,
        row alignment is the only remaining risk and the guard below covers it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def build_model(ckpt_path, arch, src_dir, device, random_init, init_seed):
    """Load a controlled-pair checkpoint, or a matched untrained baseline.

    B2 FIX. The seed MUST be set before construction. Previously manual_seed ran
    after the model was built, so --init-seed influenced nothing and PyTorch's
    per-process nondeterministic default seed was used. Because the runner invokes
    this script twice per (model, layer) -- once for --shards-dir, once for
    --fasta-dir -- the untrained arm trained its SAE on one random network and
    scored it on a DIFFERENT one. That biases the untrained floor toward zero,
    which flatters every trained arm it is compared against.
    """
    sys.path.insert(0, str(src_dir))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, meta = ck["cfg"], ck["meta"]

    if random_init:
        torch.manual_seed(init_seed)          # BEFORE construction -- see above

    if arch == "esmc":
        from model_ctrl_esmc import CtrlESMC
        model = CtrlESMC(**cfg)
    else:
        from model_ctrl_plm import PLM, PLMConfig
        model = PLM(PLMConfig(**cfg))

    if not random_init:
        model.load_state_dict(ck["model"])
    model.eval().to(device)

    # Fingerprint: two invocations of the same untrained arm must print the same
    # value. If they differ, the two embedding sets came from different networks.
    with torch.no_grad():
        w = next(model.parameters()).flatten()[:4].tolist()
    print(f"  weight fingerprint [{', '.join(f'{x:+.6f}' for x in w)}]"
          f"{'  (random_init seed=%d)' % init_seed if random_init else ''}")
    return model, cfg, meta


def token_ids(meta, arch):
    """(vocab_lookup, bos, eos, pad, unk) normalised across the two generations.

    B8 FIX. esmc meta has NO 'unk' key and aa2id holds exactly the 20 standard
    residues, so the old `meta.get("unk", meta.get("mask"))` resolved to
    <mask>=32 -- the MLM corruption token, which the causal arm never saw in
    training. The reference loader (sae_review/src/eval_ctrl_plm.py:107) uses
    aa2id["A"], so two loaders for the same backbone disagreed on the same input.
    Match the reference loader, and make every substitution loud.
    """
    if arch == "esmc":
        v = meta["aa2id"]
        unk = v.get("A")                       # reference-loader behaviour
    else:
        v = meta["vocab"]
        unk = meta.get("unk", v.get("A"))
    return v, meta["bos"], meta["eos"], meta["pad"], unk


_SUBS = {"n": 0, "trunc": 0}


def encode(seq, v, bos, eos, unk, max_len):
    s = seq.upper()                            # reference loader upper-cases
    if len(s) > max_len - 2:
        _SUBS["trunc"] += 1
    ids = [bos]
    for ch in s[: max_len - 2]:
        if ch not in v:
            _SUBS["n"] += 1
        ids.append(v.get(ch, unk))
    ids.append(eos)
    return ids


def embed(model, seqs, layer, meta, arch, max_len, bs, device):
    v, bos, eos, pad, unk = token_ids(meta, arch)
    feats, bounds, cur = [], [], 0
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            toks = [encode(s, v, bos, eos, unk, max_len) for s in seqs[i:i + bs]]
            T = max(len(t) for t in toks)
            ids = torch.full((len(toks), T), pad, dtype=torch.long)
            am = torch.zeros((len(toks), T), dtype=torch.long)
            for r, t in enumerate(toks):
                ids[r, :len(t)] = torch.tensor(t)
                am[r, :len(t)] = 1
            _, hid = model(ids.to(device), am.to(device), return_hidden=True)
            h = hid[layer].float().cpu().numpy()
            for r, t in enumerate(toks):
                L = len(t) - 2                       # strip BOS/EOS
                feats.append(h[r, 1:1 + L, :])
                bounds.append((cur, cur + L)); cur += L
    return np.concatenate(feats, 0).astype(np.float32), bounds


def meta_yaml(d, n, dim, layer, tag):
    (d / "metadata.yaml").write_text(
        f"d_model: {dim}\ndtype: float32\nlayer: {layer}\nmodel: {tag}\n"
        f"total_tokens: {n}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", choices=["esmc", "plm"], required=True)
    ap.add_argument("--src-dir", default=".",
                    help="dir containing model_ctrl_esmc.py / model_ctrl_plm.py")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--fasta-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="load model, embed 8 dummy proteins, print shapes, exit")
    args = ap.parse_args()

    model, cfg, meta = build_model(Path(args.ckpt).expanduser(), args.arch,
                                   Path(args.src_dir), args.device,
                                   args.random_init, args.init_seed)
    max_len = int(cfg.get("max_seq", 512))
    tag = args.model_tag or Path(args.ckpt).parent.name
    print(f"  arch={args.arch} d_model={cfg['d_model']} layers={cfg['n_layers']} "
          f"causal={cfg.get('causal')} max_seq={max_len} random_init={args.random_init}")
    if not (0 <= args.layer < cfg["n_layers"]):
        raise SystemExit(f"  --layer {args.layer} out of range ({cfg['n_layers']} layers)")

    if args.smoke:
        demo = ["MKTAYIAKQRQISFVKSHFSRQ", "MSKGEELFTGVVPILVELDGDV",
                "MVLSPADKTNVKAAWGKVGAHA", "MEEPQSDPSVEPPLSQETFSDL",
                "MADQLTEEQIAEFKEAFSLFDK", "MGSSHHHHHHSSGLVPRGSHM",
                "MTEYKLVVVGAGGVGKSALTI", "MSDNGPQNQRNAPRITFGGPS"]
        X, b = embed(model, demo, args.layer, meta, args.arch, max_len, 4, args.device)
        print(f"  SMOKE OK: {X.shape} rows, {len(b)} proteins, "
              f"sum(len)={sum(len(s) for s in demo)} (must equal rows)")
        return

    if not args.out:
        raise SystemExit("  --out required unless --smoke")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.fasta_dir:
        fastas = sorted(Path(args.fasta_dir).glob("shard_*.fasta"))
        if not fastas:
            raise SystemExit(f"  no shard_*.fasta under {args.fasta_dir} — "
                             f"refusing to write a .done sentinel for zero work")
        for ff in fastas:
            seqs, cur = [], []
            for line in open(ff):
                if line.startswith(">"):
                    if cur: seqs.append("".join(cur)); cur = []
                else: cur.append(line.strip())
            if cur: seqs.append("".join(cur))
            X, _ = embed(model, seqs, args.layer, meta, args.arch, max_len,
                         args.batch_size, args.device)
            sd = out / ff.stem; sd.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(X), sd / "activations.pt")
            meta_yaml(sd, X.shape[0], X.shape[1], args.layer, tag)
            print(f"  {ff.stem}: {X.shape[0]:,} x {X.shape[1]}")
        # B5: truncation on the --fasta-dir path has no alignment guard to catch
        # it (there is no concept matrix to compare against), so report it loudly.
        if _SUBS["trunc"]:
            print(f"  !! {_SUBS['trunc']} training sequences TRUNCATED at "
                  f"{max_len - 2} residues. The SAE's training distribution is "
                  f"now N-terminally biased and no longer matches the analysis "
                  f"set. Pass --max_length to subset_fasta.py instead.")
        if _SUBS["n"]:
            print(f"  !! {_SUBS['n']} non-standard residues mapped to the "
                  f"fallback token")
        (out / ".done").write_text(str(len(fastas)))   # B6 sentinel
        return

    shard_files = sorted(Path(args.shards_dir).glob("shard_*/protein_data.tsv"))
    if not shard_files:
        raise SystemExit(f"  no shard_*/protein_data.tsv under {args.shards_dir} — "
                         f"refusing to write a .done sentinel for zero work")
    for sf in shard_files:
        df = pd.read_csv(sf, sep="\t")
        col = next(c for c in df.columns if c.lower() == "sequence")
        expected = sp.load_npz(sf.parent / "aa_concepts.npz").shape[0]
        X, bounds = embed(model, df[col].astype(str).tolist(), args.layer, meta,
                          args.arch, max_len, args.batch_size, args.device)
        if X.shape[0] != expected:
            raise SystemExit(
                f"  {sf.parent.name}: ROW MISALIGNMENT — {X.shape[0]} rows vs "
                f"{expected} in aa_concepts.npz. Refusing to write. A sequence "
                f"exceeded max_seq-2={max_len - 2}; extract_annotations does not "
                f"truncate, so filter the protein set by length first.")
        sd = out / sf.parent.name; sd.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": torch.from_numpy(X), "boundaries": bounds,
                    "protein_ids": df["Entry"].tolist() if "Entry" in df.columns
                    else list(range(len(df)))}, sd / "embeddings.pt")
        meta_yaml(sd, X.shape[0], X.shape[1], args.layer, tag)
        print(f"  {sf.parent.name}: {X.shape[0]:,} x {X.shape[1]}  (aligned)")

    if _SUBS["n"]:
        print(f"  !! {_SUBS['n']} non-standard residues mapped to the fallback token")
    # B6: sentinel written ONLY after every shard succeeds. The runner must test
    # for this file, not for the directory -- out.mkdir() happens before any
    # embedding, so a killed run leaves a directory that looks complete and the
    # SAE then silently trains on a partial shard set.
    (out / ".done").write_text(str(len(shard_files)))


if __name__ == "__main__":
    main()
