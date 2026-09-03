"""Label SAE features by their highest-activation residues.

The reviewer asked for exactly this: "check whether activations 'make sense' for any
individual residue by labelling them on the basis of highest-activation examples".

For every feature we take its top-K activating residues and ask what they have in
common -- amino acid, secondary structure (DSSP collapsed to helix/strand/coil), or
SCOPe fold. Enrichment is the fraction within the top-K divided by that label's
background frequency over all scored residues, so 1.0 means "no more concentrated than
chance".

This is the qualitative check that concept-F1 and the per-feature structure correlations
do not provide, and it discriminates between the two readings currently in tension:
  concept-F1 says features align with folds/superfamilies (fold:c.66 test-F1 1.000)
  feature_interpretability.csv says they barely align with per-residue secondary
  structure (median |corr| 0.02-0.03, 5 of 2560 features with |corr| > 0.2)
If the top-K sets are fold- or amino-acid-concentrated but not SS-concentrated, those
two results are one coherent picture rather than a contradiction.
"""
import argparse
import collections
import json
import os
import re

import numpy as np

SS3 = {"H": "helix", "G": "helix", "I": "helix",
       "E": "strand", "B": "strand"}


def load_folds(fasta):
    """uid -> (fold, superfamily) from SCOPe fasta headers: '>d1dlwa_ a.1.1.1 (A:) ...'."""
    fold, sf = {}, {}
    with open(fasta) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m = re.match(r">(\S+)\s+([a-z])\.(\d+)\.(\d+)\.(\d+)", line)
            if m:
                u, cl, fo, s1, _ = m.groups()
                fold[u] = "%s.%s" % (cl, fo)
                sf[u] = "%s.%s.%s" % (cl, fo, s1)
    return fold, sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--fasta", default="cache/scope_40.fa")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.layer_dir
    Z = np.load(os.path.join(d, "Z.npy"), mmap_mode="r")
    uids = [str(u) for u in json.loads(open(os.path.join(d, "uids.json")).read())]
    lengths = np.load(os.path.join(d, "lengths.npy"))
    offsets = np.load(os.path.join(d, "offsets.npy"))
    seqs_raw = json.loads(open(os.path.join(d, "sequences.json")).read())
    seqs = seqs_raw if isinstance(seqs_raw, dict) else dict(zip(uids, seqs_raw))

    nres, nfeat = Z.shape
    print("  %s : %d residues x %d features, %d proteins" % (d, nres, nfeat, len(uids)))

    # row -> protein index, then -> (uid, position)
    prot_of_row = np.searchsorted(offsets, np.arange(nres), side="right") - 1
    pos_of_row = np.arange(nres) - offsets[prot_of_row]

    ss_map = {}
    with open(args.features_csv) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split(",")
            ss_map[(p[0], int(p[1]))] = p[2]

    fold_of, sf_of = load_folds(args.fasta)

    aa_lab = np.empty(nres, dtype="<U1")
    ss_lab = np.empty(nres, dtype="<U6")
    fo_lab = np.empty(nres, dtype="<U8")
    for i in range(nres):
        k = prot_of_row[i]
        u = uids[k]
        pos = int(pos_of_row[i])
        s = seqs.get(u, "")
        aa_lab[i] = s[pos] if pos < len(s) else "?"
        ss_lab[i] = SS3.get(ss_map.get((u, pos), "-"), "coil")
        fo_lab[i] = fold_of.get(u, "?")

    def background(lab):
        c = collections.Counter(lab.tolist())
        n = float(len(lab))
        return {k: v / n for k, v in c.items()}

    bg = {"aa": background(aa_lab), "ss": background(ss_lab), "fold": background(fo_lab)}
    labs = {"aa": aa_lab, "ss": ss_lab, "fold": fo_lab}

    rows = []
    K = args.topk
    for f in range(nfeat):
        col = np.asarray(Z[:, f], dtype=np.float32)
        if not np.isfinite(col).any() or col.max() <= 0:
            continue
        top = np.argpartition(-col, min(K, nres - 1))[:K]
        top = top[np.argsort(-col[top])]
        rec = {"feature": f, "max_act": float(col[top[0]])}
        for kind in ("aa", "ss", "fold"):
            c = collections.Counter(labs[kind][top].tolist())
            lab, cnt = c.most_common(1)[0]
            frac = cnt / float(K)
            rec["%s_top" % kind] = lab
            rec["%s_frac" % kind] = frac
            rec["%s_enrich" % kind] = frac / max(bg[kind].get(lab, 1e-9), 1e-9)
        rows.append(rec)

    print("  scored %d live features (top-%d residues each)" % (len(rows), K))
    print()
    for kind, thresh in (("aa", 0.5), ("ss", 0.8), ("fold", 0.5)):
        fr = np.array([r["%s_frac" % kind] for r in rows])
        en = np.array([r["%s_enrich" % kind] for r in rows])
        print("  %-5s dominant-label fraction: median %.3f  p90 %.3f | "
              "enrichment: median %5.1fx  p90 %6.1fx | features over %.0f%%: %d (%.1f%%)"
              % (kind, np.median(fr), np.percentile(fr, 90),
                 np.median(en), np.percentile(en, 90),
                 100 * thresh, int((fr > thresh).sum()), 100 * (fr > thresh).mean()))
    print()
    print("  (ss uses a 0.8 bar because coil alone is ~%.0f%% of residues at background)"
          % (100 * bg["ss"].get("coil", 0)))
    print("  background: %s" % {k: round(v, 3) for k, v in sorted(bg["ss"].items())})
    print()
    top_aa = sorted(rows, key=lambda r: -r["aa_frac"])[:8]
    print("  most amino-acid-pure features:")
    for r in top_aa:
        print("    f%-5d  %s %.0f%% of top-%d (%.0fx bg)   ss %s %.0f%%   fold %s %.0f%%"
              % (r["feature"], r["aa_top"], 100 * r["aa_frac"], K, r["aa_enrich"],
                 r["ss_top"], 100 * r["ss_frac"], r["fold_top"], 100 * r["fold_frac"]))
    print()
    top_fo = sorted(rows, key=lambda r: -r["fold_frac"])[:8]
    print("  most fold-pure features:")
    for r in top_fo:
        print("    f%-5d  fold %s %.0f%% of top-%d (%.0fx bg)   aa %s %.0f%%   ss %s %.0f%%"
              % (r["feature"], r["fold_top"], 100 * r["fold_frac"], K, r["fold_enrich"],
                 r["aa_top"], 100 * r["aa_frac"], r["ss_top"], 100 * r["ss_frac"]))

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print()
        print("  wrote -> %s" % args.out)


if __name__ == "__main__":
    main()
