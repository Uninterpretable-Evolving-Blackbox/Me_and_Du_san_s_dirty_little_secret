"""Third fix in experiment_raw_coactivation.py: the perplexity() model API.

perplexity() is written against a HuggingFace-style interface that CtrlESMC does not
have. Three determinate problems, all in this one function:

1. CtrlESMC.forward(input_ids, attention_mask, return_hidden=False) takes the mask as a
   REQUIRED positional argument:
       TypeError: CtrlESMC.forward() missing 1 required positional argument: 'attention_mask'
   perplexity() builds one protein at a time, batch of 1, with no padding
   (x = torch.tensor([ids])), so the mask is unambiguously all ones.

2. `model(x)["logits"] if isinstance(model(x), dict) else model(x)` calls the model
   TWICE per protein, and the dict branch is dead: CtrlESMC.forward returns a tensor.

3. mask_id = aa2id.get("<mask>", aa2id.get("X")) evaluates to None. The checkpoint's
   aa2id holds exactly the 20 standard amino acids -- no "<mask>", no "X". The real
   mask token is recorded separately by train_ctrl_plm.py as meta["mask"] (= 32 for
   these checkpoints, alongside pad=1, bos=0, eos=2). With mask_id None the masked
   (pseudo-perplexity) branch would die at `xm[0, i] = mask_id`, so the MLM arm could
   never have produced a number.

The fix threads mask_id in from meta at the call site and leaves the arithmetic alone.
"""
import ast
import sys

P = "experiment_raw_coactivation.py"
src = open(P).read()

EDITS = [
    # 1. signature: accept the mask token explicitly
    ("def perplexity(model, uids, seqs, aa2id, bos, eos, pad, device, causal, max_prot=150):",
     "def perplexity(model, uids, seqs, aa2id, bos, eos, pad, device, causal,\n"
     "               max_prot=150, mask_id=None):"),

    # 2. mask token: meta['mask'], not a lookup in the 20-letter aa2id
    ('    mask_id = aa2id.get("<mask>", aa2id.get("X"))',
     '    # aa2id holds only the 20 standard amino acids; the mask token is recorded\n'
     '    # separately as meta["mask"] by train_ctrl_plm.py. Falling back to the old\n'
     '    # lookup would give None and kill the masked branch at `xm[0, i] = mask_id`.\n'
     '    if mask_id is None:\n'
     '        mask_id = aa2id.get("<mask>", aa2id.get("X"))'),

    # 3. causal branch: pass the (all-ones) attention mask, and call the model once
    ('            logits = model(x)["logits"] if isinstance(model(x), dict) else model(x)',
     '            # batch of 1, no padding -> the attention mask is all ones.\n'
     '            # CtrlESMC.forward returns a tensor, so the old dict branch was dead\n'
     '            # code that also ran the model twice per protein.\n'
     '            logits = model(x, torch.ones_like(x))'),

    # 4. masked branch: same
    ('                out = model(xm)',
     '                out = model(xm, torch.ones_like(xm))'),

    # 5. call site: hand it the mask token from the checkpoint meta
    ('            ppl, ppl_n = perplexity(model, uids, seqs, aa2id, bos, eos, pad,\n'
     '                                    device, causal)',
     '            ppl, ppl_n = perplexity(model, uids, seqs, aa2id, bos, eos, pad,\n'
     '                                    device, causal, mask_id=meta.get("mask"))'),
]

for i, (old, _new) in enumerate(EDITS, 1):
    if old not in src:
        sys.exit("FAIL: edit %d target not found verbatim -- aborting without writing" % i)

for old, new in EDITS:
    src = src.replace(old, new)

ast.parse(src)
open(P, "w").write(src)
print("patched %d sites; syntax OK" % len(EDITS))
