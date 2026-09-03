"""Encode ESM-2 layer-18 activations with InterPLM's own released SAE.

THE QUESTION. In LLMs, single-token SAE features are a rare early-layer phenomenon:
1.48% of features in GPT2-Small (91% of them in layer 0), 0.47% in Gemma-2-2B, 0.14% in
Gemma-2-9B, near-zero in LlamaScope-8B (arXiv 2607.20596). In our 42M protein models they
are the dominant mode at mid-depth: the median masked-arm feature's top-50 residues are
100% one amino acid, and 78.6% of features exceed 50% purity at L11 of 30.

That gap is either a DOMAIN effect (20 tokens with strong biophysical correlates, versus
50k text tokens) or a SCALE effect (42M versus billions). This settles it, on the exact
objects the field uses: ESM-2-650M with InterPLM's published SAE.

ARCHITECTURE, verified rather than assumed. The state dict holds
{bias, encoder.weight, encoder.bias, decoder.weight}, which is interplm's ReLUSAE
(TopKSAE names its pre-bias `b_dec`). Its encode is, verbatim from
interplm/sae/dictionary.py:

    features = nn.ReLU()(self.encoder(x - self.bias))

The README says the `ae_normalized.pt` variant "activate[s] between 0-1 based on max
activation values from Swiss-Prot", but no `activation_rescale_factor` buffer is present
in the checkpoint, so whether the rescaling is folded into the weights is unverified. The
script therefore REPORTS the observed activation scale instead of assuming it -- if the
per-feature maxima sit near 1.0 the normalisation is baked in, and InterPLM's ">0.6" gate
is directly comparable.

Capacity matches our own configuration (expansion 8x: 1280 -> 10240, versus our
320 -> 2560), so the comparison is not confounded by dictionary size.
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True,
                    help="dir with raw_embeddings.npy + uids.json + lengths.npy")
    ap.add_argument("--repo", default="Elana/InterPLM-esm2-650m")
    ap.add_argument("--sae-layer", type=int, default=18)
    ap.add_argument("--variant", default="ae_normalized.pt")
    ap.add_argument("--batch", type=int, default=32768)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import hf_hub_download

    d = args.layer_dir
    X = np.load(os.path.join(d, "raw_embeddings.npy"), mmap_mode="r")
    n, dim = X.shape
    print("  activations: %s from %s" % ((n, dim), d))

    sd = torch.load(hf_hub_download(args.repo, "layer_%d/%s" % (args.sae_layer, args.variant)),
                    map_location="cpu", weights_only=False)
    W_enc = sd["encoder.weight"].float()      # (F, D)
    b_enc = sd["encoder.bias"].float()        # (F,)
    b_pre = sd["bias"].float()                # (D,)  subtracted before the encoder
    F = W_enc.shape[0]
    assert W_enc.shape[1] == dim, "dim mismatch: SAE %d vs activations %d" % (W_enc.shape[1], dim)
    print("  SAE: %d features, expansion %.1fx" % (F, F / dim))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W_enc, b_enc, b_pre = W_enc.to(dev), b_enc.to(dev), b_pre.to(dev)

    Z = np.lib.format.open_memmap(os.path.join(d, "Z.npy"), mode="w+",
                                  dtype=np.float16, shape=(n, F))
    fmax = torch.zeros(F, device=dev)
    nz = 0
    with torch.no_grad():
        for i in range(0, n, args.batch):
            xb = torch.from_numpy(np.asarray(X[i:i + args.batch])).float().to(dev)
            f = torch.relu((xb - b_pre) @ W_enc.T + b_enc)
            fmax = torch.maximum(fmax, f.max(dim=0).values)
            nz += int((f > 0).sum().item())
            Z[i:i + args.batch] = f.cpu().numpy().astype(np.float16)
            if (i // args.batch) % 3 == 0:
                print("    %d/%d rows" % (min(i + args.batch, n), n), flush=True)
    Z.flush()

    fm = fmax.cpu().numpy()
    print()
    print("  wrote %s  (%.1f GB)" % (os.path.join(d, "Z.npy"), Z.nbytes / 1073741824))
    print("  sparsity: %.4f of entries nonzero" % (nz / float(n) / F))
    print("  per-feature max: median %.3f  p90 %.3f  max %.3f  |  dead (max==0): %d/%d"
          % (np.median(fm), np.percentile(fm, 90), fm.max(), int((fm == 0).sum()), F))
    near1 = float(((fm > 0.5) & (fm < 2.0)).mean())
    print("  fraction of features with max in [0.5, 2.0]: %.3f" % near1)
    print("  -> if that is high, the Swiss-Prot normalisation is folded into the weights")
    print("     and InterPLM's absolute > 0.6 gate applies directly to these values.")
    json.dump({"n_features": int(F), "median_feature_max": float(np.median(fm)),
               "max_feature_max": float(fm.max()), "dead": int((fm == 0).sum()),
               "sparsity": nz / float(n) / F},
              open(os.path.join(d, "sae_encode_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
