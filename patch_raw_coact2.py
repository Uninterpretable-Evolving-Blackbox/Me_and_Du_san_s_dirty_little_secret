"""Second fix in experiment_raw_coactivation.py: load_eval_set's real signature.

eval_ctrl_plm.load_eval_set(d) returns THREE values -- (uids, uid2seq, val_uids).
experiment_raw_coactivation.py unpacks SIX, expecting the tokenizer fields to come
back with them:

    uids, seqs, aa2id, bos, eos, pad = load_eval_set(args.eval_set)
    ValueError: not enough values to unpack (expected 6, got 3)

The tokenizer fields live in the checkpoint's "meta", not in the eval set.
eval_ctrl_plm.py:181-182 is the reference:

    extract_layer(model, uids, seqs, args.layer, meta["aa2id"],
                  meta["bos"], meta["eos"], meta["pad"], dev)

meta carries pad=1, bos=0, eos=2 and a 20-entry aa2id for these checkpoints.
The downstream code already treats `seqs` as a uid->sequence dict, which is exactly
what load_eval_set's second return value is, so nothing else changes.
"""
import ast
import sys

P = "experiment_raw_coactivation.py"
src = open(P).read()

OLD = '        uids, seqs, aa2id, bos, eos, pad = load_eval_set(args.eval_set)'
NEW = (
    '        # load_eval_set returns (uids, uid2seq, val_uids) -- three values. The\n'
    '        # tokenizer fields come from the checkpoint meta, as in eval_ctrl_plm.py:181.\n'
    '        uids, seqs, _val_uids = load_eval_set(args.eval_set)\n'
    '        meta = ck["meta"]\n'
    '        aa2id, bos, eos, pad = (meta["aa2id"], meta["bos"],\n'
    '                                meta["eos"], meta["pad"])'
)

if OLD not in src:
    sys.exit("FAIL: target line not found verbatim -- aborting without writing")

src = src.replace(OLD, NEW)
ast.parse(src)
open(P, "w").write(src)
print("patched load_eval_set unpacking; syntax OK")
