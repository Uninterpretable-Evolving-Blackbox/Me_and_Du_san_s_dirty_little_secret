"""Fourth fix in experiment_raw_coactivation.py: extract_layer's contract.

Two problems on adjacent lines, both from calling eval_ctrl_plm.extract_layer with the
wrong types.

(a) `seqs` must be a LIST aligned to uids, not the uid->seq dict.
    extract_layer indexes it positionally:
        chunk = list(range(i, min(i + batch_size, len(uids))))
        toks  = [... for a in seqs[j].upper() ...]      # j is an int
    Passing the dict gives `KeyError: 0`. Both working callers build the list first --
    eval_ctrl_plm.py:160 and eval_ctrl_saefree.py:252 are identical:
        seqs = [uid2seq[u] for u in uids]
    The dict is still needed here by perplexity() and ref_seqs, so keep both.

(b) extract_layer returns a 2-TUPLE (X, lengths); the code binds it to X alone and then
    calls np.asarray on it.

Using extract_layer's returned lengths (rather than len(seq) from the dict) also fixes a
latent misalignment: extract_layer truncates at max_len=512 and reports L = len(t) - 2,
so for any protein longer than 510 residues the precomputed lengths would not describe
the rows of X that struct_delta_on_matrix is about to consume.
"""
import ast
import sys

P = "experiment_raw_coactivation.py"
src = open(P).read()

EDITS = [
    ('        uids = [str(u) for u in uids]\n'
     '        ref_seqs = {u: seqs[u] for u in uids}',
     '        uids = [str(u) for u in uids]\n'
     '        # extract_layer indexes sequences positionally, so it needs a list aligned\n'
     '        # to uids (eval_ctrl_plm.py:160, eval_ctrl_saefree.py:252). perplexity() and\n'
     '        # ref_seqs below still want the dict, so keep both.\n'
     '        seq_list = [seqs[u] for u in uids]\n'
     '        ref_seqs = {u: seqs[u] for u in uids}'),

    ('            X = extract_layer(model, uids, seqs, L, aa2id, bos, eos, pad, device)',
     '            # extract_layer returns (X, lengths). Its lengths are the ones that\n'
     '            # describe X: it truncates at max_len=512, so they can differ from\n'
     '            # len(seq) for long proteins.\n'
     '            X, lengths = extract_layer(model, uids, seq_list, L,\n'
     '                                       aa2id, bos, eos, pad, device)'),
]

for i, (old, _new) in enumerate(EDITS, 1):
    if old not in src:
        sys.exit("FAIL: edit %d target not found verbatim -- aborting without writing" % i)

for old, new in EDITS:
    src = src.replace(old, new)

ast.parse(src)
open(P, "w").write(src)
print("patched %d sites; syntax OK" % len(EDITS))
