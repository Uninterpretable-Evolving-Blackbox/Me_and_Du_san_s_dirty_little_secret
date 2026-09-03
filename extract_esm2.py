"""Extract ESM-2 layer-16 per-residue states for the stage-11 positive control.

WHICH ESM-2. The repo pins it implicitly rather than by name:
  crosscoder.py:42-44   "k/embed = 256/1280 = 20.0% (ESM-2)"        -> d_model 1280
  compute_h1_bootstrap.py:99  "the paper's ESM-2 grid is index/32: L0..L32"  -> 33 blocks
d_model 1280 with 33 blocks is facebook/esm2_t33_650M_UR50D, and layer 16 of 32 is the
50% depth that RITA's layer 12 of 24 matches in experiment_aa_selectivity.py's defaults.

LAYER INDEX AMBIGUITY, handled rather than guessed. HuggingFace returns 34 hidden states
for a 33-block model: index 0 is the embedding output, 1..33 are block outputs. The
project's own extractor (eval_ctrl_plm.extract_layer) indexes the *block-output* list, so
"L16" should be hidden_states[17]. But the paper's L0..L32 grid could equally be
hidden_states[0..32]. Adjacent layers are highly similar, so both are written and the
probe is run on each; whichever reproduces the reported 0.606 settles it, and the
disagreement between them bounds how much the choice matters.

ROW ALIGNMENT. uids.json / lengths.npy / sequences.json are copied verbatim from
outputs_ctrl/ckpt_clm_s42/layer_14 so the rows line up with everything else in the
project: 1500 proteins, 293,760 residues, sequences truncated at 510.
"""
import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--align-dir", required=True,
                    help="dir holding uids.json / lengths.npy / sequences.json")
    ap.add_argument("--out-root", default="outputs_layerwise/esm2")
    ap.add_argument("--layers", default="16,17",
                    help="hidden_states indices to save (see docstring)")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel

    uids = [str(u) for u in json.loads(open(os.path.join(args.align_dir, "uids.json")).read())]
    lengths = np.load(os.path.join(args.align_dir, "lengths.npy"))
    seqs_raw = json.loads(open(os.path.join(args.align_dir, "sequences.json")).read())
    seqs = seqs_raw if isinstance(seqs_raw, dict) else dict(zip(uids, seqs_raw))
    total = int(lengths.sum())
    print("  %d proteins, %d residues, max len %d" % (len(uids), total, int(lengths.max())))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval().cuda()
    d = model.config.hidden_size
    nblocks = model.config.num_hidden_layers
    print("  %s : d_model %d, %d blocks -> %d hidden_states"
          % (args.model, d, nblocks, nblocks + 1))

    want = [int(x) for x in args.layers.split(",")]
    buf = {L: np.zeros((total, d), dtype=np.float32) for L in want}

    row = 0
    with torch.no_grad():
        for i in range(0, len(uids), args.batch_size):
            chunk = list(range(i, min(i + args.batch_size, len(uids))))
            batch = [seqs[uids[j]][:int(lengths[j])] for j in chunk]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.cuda() for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            for bi, j in enumerate(chunk):
                L = int(lengths[j])
                for Lx in want:
                    # token 0 is <cls>; residues are tokens 1..L
                    buf[Lx][row:row + L] = hs[Lx][bi, 1:1 + L].float().cpu().numpy()
                row += L
            if (i // args.batch_size) % 25 == 0:
                print("    %d/%d proteins, row %d/%d" % (chunk[-1] + 1, len(uids), row, total),
                      flush=True)
    assert row == total, "row mismatch: %d vs %d" % (row, total)

    import shutil
    for Lx in want:
        outd = os.path.join(args.out_root, "layer_%d" % Lx)
        os.makedirs(outd, exist_ok=True)
        np.save(os.path.join(outd, "raw_embeddings.npy"), buf[Lx])
        for f in ("uids.json", "lengths.npy", "sequences.json"):
            shutil.copy(os.path.join(args.align_dir, f), os.path.join(outd, f))
        print("  wrote %s : %s" % (outd, buf[Lx].shape))


if __name__ == "__main__":
    main()
