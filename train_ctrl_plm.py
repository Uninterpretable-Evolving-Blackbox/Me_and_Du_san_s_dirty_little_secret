#!/usr/bin/env python3
"""
train_ctrl_plm.py — train ONE objective (mlm|clm) on the shared corpus.

Both objectives use the SAME corpus, the SAME data/batch order (--data-order-seed),
the SAME architecture and the SAME init seed; only --objective (attention mask + loss
construction) differs. That is what isolates the objective. bf16 mixed precision on MPS,
AdamW + warmup/cosine.

Usage:
  python train_ctrl_plm.py --objective mlm --data-dir ~/own_sae_data/uniref50_smoke --smoke
  python train_ctrl_plm.py --objective clm --data-dir ~/own_sae_data/uniref50_smoke --smoke
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model_ctrl_esmc import CtrlESMC, ESMC_DEPTH


def load_corpus(d):
    d = Path(d)
    tokens = np.load(d / "tokens.npy")
    lengths = np.load(d / "lengths.npy")
    offsets = np.load(d / "offsets.npy")
    meta = json.loads((d / "meta.json").read_text())
    return tokens, lengths, offsets, meta


class SeqData:
    def __init__(self, tokens, lengths, offsets, idx):
        self.tokens, self.lengths, self.offsets, self.idx = tokens, lengths, offsets, idx

    def __len__(self):
        return len(self.idx)

    def get(self, i):
        j = self.idx[i]
        o = int(self.offsets[j]); l = int(self.lengths[j])
        return self.tokens[o:o + l].astype(np.int64)


def make_batches(n, bs, seed, steps):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    batches, p = [], 0
    while len(batches) < steps:
        if p + bs > len(order):
            order = rng.permutation(n); p = 0
        batches.append(order[p:p + bs]); p += bs
    return batches


def collate(seqs, pad, seq_len):
    B = len(seqs)
    T = min(seq_len, max(len(s) for s in seqs))
    ids = np.full((B, T), pad, dtype=np.int64)
    am = np.zeros((B, T), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:T]
        ids[i, :len(s)] = s
        am[i, :len(s)] = 1
    return torch.from_numpy(ids), torch.from_numpy(am)


def mlm_corrupt(ids, am, mask_id, aa_lo, aa_hi, rate, specials):
    """Dynamic MLM masking (RoBERTa/ESM-style). Vectorised. Returns (input, labels)."""
    prob = torch.full(ids.shape, float(rate), device=ids.device)
    special = (am == 0)
    for sid in specials:
        special = special | (ids == sid)
    prob[special] = 0.0
    sel = torch.bernoulli(prob).bool()
    labels = torch.where(sel, ids, torch.full_like(ids, -100))
    inp = ids.clone()
    r = torch.rand(ids.shape, device=ids.device)
    inp[sel & (r < 0.8)] = mask_id
    randpos = sel & (r >= 0.8) & (r < 0.9)
    rand_aa = torch.randint(aa_lo, aa_hi + 1, ids.shape, device=ids.device)
    inp[randpos] = rand_aa[randpos]
    return inp, labels


def clm_shift(ids, am):
    """Next-token labels; ignore last position, pad, and positions predicting a pad."""
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    labels[am == 0] = -100
    nextpad = torch.zeros_like(am)
    nextpad[:, :-1] = am[:, 1:]
    labels[nextpad == 0] = -100
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "own_sae_data" / "uniref50_pilot"))
    ap.add_argument("--objective", choices=["mlm", "clm"], required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--target-tokens", type=float, default=700e6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-order-seed", type=int, default=1234)  # SAME for both objectives
    # ESM-C anchor: head_dim must stay 64 and depth 30 (see model_ctrl_esmc.py docstring).
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--n-heads", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=ESMC_DEPTH)
    ap.add_argument("--match-predictions", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    # --- token-ablation support -------------------------------------------------
    # A token ablation is ONE long run per arm with checkpoints along the way, not N
    # separate runs, so the whole rank-vs-tokens curve costs the same as its longest
    # point. Milestone checkpoints are model-only (~1/3 the size) since they are for
    # analysis, never for resuming.
    ap.add_argument("--ckpt-at-tokens", default="",
                    help="comma-separated token milestones, e.g. 0.66e9,2.1e9,4.2e9. "
                         "Saves model-only model_tok<N>M.pt at each (idempotent).")
    ap.add_argument("--rolling-resume", action="store_true",
                    help="periodic checkpoints overwrite ONE model_resume.pt instead of "
                         "accumulating model_step*.pt (bounds disk on multi-billion-token runs)")
    # ----------------------------------------------------------------------------
    ap.add_argument("--ckpt-every", type=int, default=5000,
                    help="save a checkpoint every N steps (~82M tokens at defaults)")
    ap.add_argument("--stop-at-tokens", type=float, default=0,
                    help="stop THIS run after N tokens; LR schedule still targets "
                         "--target-tokens, so --resume continues cleanly to the full budget")
    ap.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # Device: CUDA (RTX/datacentre) > MPS (Apple) > CPU. Nothing else about the
    # recipe changes with device — see README "Do not change the recipe".
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    if dev == "cpu":
        print("WARNING: no GPU found — this will take DAYS on CPU. Fix your torch/CUDA install.")
    tokens, lengths, offsets, meta = load_corpus(args.data_dir)
    n = len(lengths)

    rng_val = np.random.default_rng(999)
    perm = rng_val.permutation(n)
    n_val = min(2000, n // 20) or 1
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = SeqData(tokens, lengths, offsets, tr_idx)
    va = SeqData(tokens, lengths, offsets, val_idx)

    toks_per_step = args.batch_size * args.seq_len
    steps = int(args.target_tokens / toks_per_step)
    if args.objective == "mlm" and args.match_predictions:
        steps = int(steps / args.mask_rate)   # match #predictions instead of #sequences
    if args.smoke:
        steps = 30
    batches = make_batches(len(tr), args.batch_size, args.data_order_seed, steps)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = dict(vocab_size=meta["vocab_size"], d_model=args.d_model, n_heads=args.n_heads,
               n_layers=args.n_layers, causal=(args.objective == "clm"))
    model = CtrlESMC(**cfg).to(dev)
    gpu = f" | {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""
    print(f"{args.objective.upper()} | {model.num_params()/1e6:.1f}M params | dev {dev}{gpu} | "
          f"seed {args.seed} | steps {steps} | ~{steps*toks_per_step/1e6:.0f}M tok")

    # exclude 1-D params (RMSNorm weights) from weight decay — common pitfall otherwise
    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    print(f"  weight-decay groups: {sum(p.numel() for p in decay)/1e6:.1f}M decayed, "
          f"{sum(p.numel() for p in nodecay)} no-decay (norms)")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        p = (s - args.warmup) / max(1, steps - args.warmup)
        return 0.1 * args.lr + 0.5 * (0.9 * args.lr) * (1 + math.cos(math.pi * min(1.0, p)))

    mask_id = meta["mask"]
    specials = {meta["pad"], meta["bos"], meta["eos"], meta["mask"]}
    # ESM-C residue ids are contiguous (4..23); random-replacement samples from that range.
    aa_lo, aa_hi = meta["aa_lo"], meta["aa_hi"]

    def batch_loss(bidx, data):
        seqs = [data.get(i) for i in bidx]
        ids, am = collate(seqs, meta["pad"], args.seq_len)
        ids, am = ids.to(dev), am.to(dev)
        if args.objective == "mlm":
            inp, lab = mlm_corrupt(ids, am, mask_id, aa_lo, aa_hi, args.mask_rate, specials)
        else:
            inp, lab = clm_shift(ids, am)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16):
            logits = model(inp, am)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               lab.reshape(-1), ignore_index=-100)
        return loss

    out = Path(args.out_dir or (Path(args.data_dir) / f"ckpt_{args.objective}"))
    out.mkdir(parents=True, exist_ok=True)

    def save_ckpt(step, tag, model_only=False):
        d = {"model": model.state_dict(), "cfg": cfg, "meta": meta,
             "objective": args.objective, "step": step,
             "tokens": int((step + 1) * toks_per_step)}
        if not model_only:                     # optimizer state only needed for resume
            d["opt"] = opt.state_dict()
        torch.save(d, out / f"model_{tag}.pt")

    milestones = sorted(int(float(x)) for x in args.ckpt_at_tokens.split(",") if x.strip())
    if milestones:
        print(f"  token milestones: {[f'{m/1e9:.2f}B' for m in milestones]} "
              f"-> model_tok<N>M.pt (model-only)")

    def _ms_tag(ms):
        # %g keeps significant figures, so sub-1M milestones don't collide into "tok0M"
        return f"tok{ms/1e6:g}M"

    def save_milestones(step):
        """Save any milestone reached. Idempotent: skips files that already exist,
        so resuming a long run never re-writes or double-counts."""
        tok_now = (step + 1) * toks_per_step
        for ms in milestones:
            f = out / f"model_{_ms_tag(ms)}.pt"
            if tok_now >= ms and not f.exists():
                save_ckpt(step, _ms_tag(ms), model_only=True)
                print(f"  [milestone] {f.name}  (~{tok_now/1e9:.2f}B tok)", flush=True)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            # milestone checkpoints are model-only. Resuming from one silently restarts
            # Adam's moments, which is a real (invisible) change to the optimisation.
            print("  WARNING: checkpoint has no optimizer state (this looks like a "
                  "model-only milestone). Adam moments are being RESET. For a clean "
                  "resume use model_resume.pt / model_partial.pt instead.", flush=True)
        start_step = int(ck.get("step", 0)) + 1
        print(f"resumed from {args.resume} at step {start_step} "
              f"(~{start_step*toks_per_step/1e6:.0f}M tok). LR schedule still planned for {steps} steps.")

    run_to = steps
    if args.stop_at_tokens and args.stop_at_tokens > 0:
        run_to = min(steps, int(args.stop_at_tokens / toks_per_step))
        print(f"  THIS RUN stops at step {run_to} (~{run_to*toks_per_step/1e6:.0f}M tok); "
              f"full schedule = {steps} steps (~{steps*toks_per_step/1e6:.0f}M tok)")

    model.train()
    t0 = time.time()
    loss = None
    for step in range(start_step, run_to):
        bidx = batches[step]
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = batch_loss(bidx, tr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            # rate measured over THIS run only (t0 restarts on resume), so ETA stays honest
            el = max(time.time() - t0, 1e-9)
            tps = (step - start_step + 1) * toks_per_step / el
            eta_h = (run_to - step - 1) * toks_per_step / max(tps, 1e-9) / 3600
            print(f"  step {step}/{steps} loss {loss.item():.4f} lr {lr_at(step):.2e} "
                  f"{(step+1)*toks_per_step/1e6:.1f}M tok {el:.0f}s | "
                  f"{tps/1e3:.1f}k tok/s | ETA {eta_h:.1f}h")
        if args.val_every and step > 0 and step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                vl = [batch_loss(vb, va).item()
                      for vb in make_batches(len(va), args.batch_size, 7, 10)]
            print(f"  [val] step {step} loss {np.mean(vl):.4f}")
            model.train()
        save_milestones(step)
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            tag = "resume" if args.rolling_resume else f"step{step}"
            save_ckpt(step, tag)
            print(f"  [ckpt] model_{tag}.pt (~{(step+1)*toks_per_step/1e6:.0f}M tok)")

    done = run_to >= steps
    save_milestones(run_to - 1)          # catch a milestone landing on the final step
    tag = "final" if done else "partial"
    save_ckpt(run_to - 1, tag)
    fl = f"{loss.item():.4f}" if loss is not None else "n/a"
    print(f"saved -> {out/('model_'+tag+'.pt')}  (train loss {fl})")
    if not done:
        print(f"PARTIAL run: reached ~{run_to*toks_per_step/1e6:.0f}M tok. "
              f"To finish to the full {steps*toks_per_step/1e6:.0f}M-token budget:\n"
              f"  python train_ctrl_plm.py --objective {args.objective} "
              f"--data-dir {args.data_dir} --resume {out/'model_partial.pt'}")


if __name__ == "__main__":
    main()
