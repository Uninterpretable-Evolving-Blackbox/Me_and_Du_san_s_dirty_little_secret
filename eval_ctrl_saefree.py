#!/usr/bin/env python3
"""
eval_ctrl_saefree.py — SAE-FREE confirmation for the controlled MLM-vs-CLM pair.

WHY THIS MATTERS MORE THAN IT LOOKS
  A randomly initialised, FROZEN dictionary reproduces the ESM-2 > RITA effect at ~103%.
  So the SAE's learning is not what produces the result, and the honest claim is about
  representational GEOMETRY rather than learned interpretable features. The evidence
  carrying that claim is the SAE-free leg: linear probes and unsupervised contact
  prediction straight off the raw activations.

  That leg exists only for the family pair (ESM-2 / RITA), which is CONFOUNDED at the
  family level. The controlled pair — the one comparison where the objective is causally
  identified — has no SAE-free confirmation at all. The strongest claim currently rests
  on the weakest design. This closes that gap.

  It touches no SAE at any point, so it is also immune to the k_sparse=256-on-d=320
  capacity problem that the sweep exists to resolve. Its answer stands whatever the
  sweep decides — which is why it is worth running early rather than late.

SELF-CONTAINED BY DESIGN
  The readouts are reimplemented here rather than imported, because this repo does not
  carry experiment_probe_baseline.py / experiment_activation_clamping.py and dragging in
  the family-pair extraction machinery would pull ESM-2 and ProtT5 dependencies onto a
  machine that does not need them. The definitions below are matched to those files:

    * contacts        Ca-Ca < 8 A AND |i-j| >= 12  — byte-identical to L_struct's, so the
                      SAE-free readout is scored on the same contact set as the metric
                      it is corroborating (Marks 2011 / Morcos 2011).
    * contact scores  APC-corrected cosine outer product. APC is Dunn, Wahl & Gloor 2008
                      (Bioinformatics); using it on an EMBEDDING outer product as an
                      unsupervised contact readout is Rao et al. 2020. Note it removes
                      per-residue background similarity here, NOT phylogenetic background
                      — there is no alignment anywhere in this pipeline.
    * precision       top-L/5 long-range (Rao et al. 2021), with the project's owned
                      deviation of |i-j| >= 12 rather than >= 24.
    * probe           logistic regression, balanced class weights, on raw activations.

CIRCULARITY GUARD
  The probe is fit on the TRAINING proteins and scored only on the held-out ones, using
  the val split recorded in eval_set/META.json — the same split as every other result.
  The contact readout is unsupervised and is likewise reported on held-out proteins only.

Usage
  python eval_ctrl_saefree.py \
      --ckpt-root ~/own_sae_data/uniref50_pilot \
      --arms ckpt_mlm_s42_token,ckpt_clm_s42,ckpt_mlm_s43_token,ckpt_clm_s43 \
      --depths 0,4,7,11,14,18,22,26,29
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from model_ctrl_esmc import CtrlESMC
from eval_ctrl_plm import load_eval_set, extract_layer, pick_device

DEPTH_LABELS = {0: "0%", 4: "13%", 7: "25%", 11: "38%", 14: "50%",
                18: "63%", 22: "75%", 26: "88%", 29: "100%"}


# ---------------------------------------------------------------- contact readout
def predict_contacts_from_embeddings(emb):
    """APC-corrected cosine outer product. Matches experiment_activation_clamping."""
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    S = (emb / n) @ (emb / n).T
    row, col, g = S.mean(1, keepdims=True), S.mean(0, keepdims=True), S.mean()
    return S - (row * col) / (g + 1e-8)


def top_l_precision(scores, contacts, seq_gap=12, fraction=0.2):
    L = scores.shape[0]
    k = max(1, int(L * fraction))
    iu = np.triu_indices(L, k=seq_gap)
    if iu[0].size == 0:
        return np.nan
    vals, truth = scores[iu], contacts[iu]
    k = min(k, vals.size)
    top = np.argpartition(-vals, k - 1)[:k]
    return float(truth[top].mean())


def ca_coords_for(pdb_path, ref_seq):
    """Ca coordinates aligned to ref_seq, NaN where unresolved. None on failure."""
    from Bio.PDB import PDBParser
    from Bio import pairwise2
    aa3 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
           "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
           "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
           "TRP": "W", "TYR": "Y", "SEC": "U", "PYL": "O"}
    try:
        st = PDBParser(QUIET=True).get_structure("t", str(pdb_path))
    except Exception:
        return None
    best_score, best = None, None
    for model in st:
        for chain in model:
            seq, xyz = [], []
            for res in chain:
                if res.get_resname() in aa3 and "CA" in res:
                    seq.append(aa3[res.get_resname()])
                    xyz.append(res["CA"].get_coord())
            if len(seq) < 10:
                continue
            aln = pairwise2.align.globalms(ref_seq, "".join(seq),
                                           2, -1, -5, -0.5, one_alignment_only=True)
            if not aln:
                continue
            a = aln[0]
            out = np.full((len(ref_seq), 3), np.nan, dtype=np.float32)
            ri = ci = -1
            for x, y in zip(a.seqA, a.seqB):
                if x != "-":
                    ri += 1
                if y != "-":
                    ci += 1
                if x != "-" and y != "-" and 0 <= ri < len(ref_seq) and 0 <= ci < len(xyz):
                    out[ri] = xyz[ci]
            if best_score is None or a.score > best_score:
                best_score, best = a.score, out
        break  # first model only, as elsewhere in this project
    return best


def true_contacts(coords, cutoff=8.0, seq_gap=12):
    from scipy.spatial import KDTree
    L = coords.shape[0]
    cmap = np.zeros((L, L), dtype=bool)
    valid = ~np.isnan(coords).any(axis=1)
    idx = np.where(valid)[0]
    if idx.size < 2:
        return cmap
    for a, b in KDTree(coords[valid]).query_pairs(r=cutoff):
        ra, rb = int(idx[a]), int(idx[b])
        if abs(ra - rb) >= seq_gap:
            cmap[ra, rb] = cmap[rb, ra] = True
    return cmap


# ---------------------------------------------------------------- linear probe
def build_labels(uids, lengths, features_csv):
    """Per-residue helix / strand / burial labels from cache/residue_features.csv."""
    df = pd.read_csv(features_csv)
    total = int(np.sum(lengths))
    helix = np.zeros(total, np.int8)
    strand = np.zeros(total, np.int8)
    burial_raw = np.full(total, np.nan, np.float32)
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    by_uid = {str(k): v for k, v in df.groupby(df.columns[0])}
    for i, (uid, L) in enumerate(zip(uids, lengths)):
        g = by_uid.get(str(uid))
        if g is None:
            continue
        pos = g["position"].to_numpy()
        ok = (pos >= 0) & (pos < int(L))
        gi = offs[i] + pos[ok]
        ss = g["ss_8class"].to_numpy()[ok].astype(str)
        helix[gi] = np.isin(ss, ["H", "G", "I"]).astype(np.int8)
        strand[gi] = np.isin(ss, ["E", "B"]).astype(np.int8)
        if "neighbor_count" in g.columns:
            burial_raw[gi] = g["neighbor_count"].to_numpy()[ok].astype(np.float32)
    med = np.nanmedian(burial_raw)
    return {"helix": helix, "strand": strand,
            "burial": (burial_raw > med).astype(np.int8),
            "valid_burial": ~np.isnan(burial_raw)}


def run_probe(X_tr, y_tr, X_te, y_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    if y_tr.sum() < 5 or (len(y_tr) - y_tr.sum()) < 5:
        return np.nan, np.nan
    clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced", solver="liblinear")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    try:
        auc = (roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
               if len(np.unique(y_te)) > 1 else np.nan)
    except Exception:
        auc = np.nan
    return float(f1_score(y_te, pred)), float(auc)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=str(Path.home() / "own_sae_data" / "uniref50_pilot"))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--depths", default="0,4,7,11,14,18,22,26,29")
    ap.add_argument("--features-csv", default="cache/residue_features.csv")
    ap.add_argument("--pdb-dir", default="cache/pdb_files")
    ap.add_argument("--out", default="results_ctrl_saefree")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--skip-contacts", action="store_true")
    args = ap.parse_args()

    dev = pick_device(args.device)
    uids, uid2seq, val_uids = load_eval_set(args.eval_set)
    uids = [u for u in uids if u in uid2seq]
    if args.max_proteins:
        uids = uids[:args.max_proteins]
    seqs = [uid2seq[u] for u in uids]
    is_val = np.array([u in val_uids for u in uids], dtype=bool)
    if not is_val.any():
        raise SystemExit("no held-out proteins in eval_set/META.json val_uids")
    print(f"proteins {len(uids)} | held-out {int(is_val.sum())} | device {dev}")

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)
    rows = []

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        ck_path = Path(args.ckpt_root) / arm / "model_final.pt"
        if not ck_path.exists():
            print(f"  [skip] no {ck_path}")
            continue
        ck = torch.load(str(ck_path), map_location="cpu")
        cfg, meta = ck["cfg"], ck["meta"]
        model = CtrlESMC(**cfg)
        model.load_state_dict(ck["model"])
        model = model.to(dev)
        print(f"\n=== {arm} | causal={cfg['causal']} | d_model={cfg['d_model']} "
              f"| blocks={cfg['n_layers']} ===")

        for layer in depths:
            X, lengths = extract_layer(model, uids, seqs, layer, meta["aa2id"],
                                       meta["bos"], meta["eos"], meta["pad"], dev)
            lengths = np.asarray(lengths, dtype=np.int64)
            offs = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
            rec = dict(arm=arm, causal=bool(cfg["causal"]), layer=layer,
                       rel_depth=DEPTH_LABELS.get(layer, f"L{layer}"))

            # residue-level split derived from the PROTEIN split (no leakage)
            res_val = np.zeros(int(lengths.sum()), dtype=bool)
            for i, (L, v) in enumerate(zip(lengths, is_val)):
                if v:
                    res_val[offs[i]:offs[i] + int(L)] = True

            labels = build_labels(uids, lengths, args.features_csv)
            for task in ("helix", "strand", "burial"):
                y = labels[task].astype(int)
                vm = labels["valid_burial"] if task == "burial" else np.ones(len(y), bool)
                tr, te = (~res_val) & vm, res_val & vm
                if tr.sum() < 50 or te.sum() < 50:
                    rec[f"probe_{task}_f1"] = np.nan
                    continue
                f1, auc = run_probe(X[tr], y[tr], X[te], y[te])
                rec[f"probe_{task}_f1"], rec[f"probe_{task}_auroc"] = f1, auc

            if not args.skip_contacts:
                precs = []
                for i, (uid, L) in enumerate(zip(uids, lengths)):
                    if not is_val[i]:
                        continue
                    p = pdb_dir / f"{str(uid)[1:5].lower()}.pdb"
                    if not p.exists():
                        continue
                    coords = ca_coords_for(p, uid2seq[uid])
                    if coords is None or coords.shape[0] != int(L):
                        continue
                    tc = true_contacts(coords)
                    if tc.sum() < 1:
                        continue
                    precs.append(top_l_precision(
                        predict_contacts_from_embeddings(X[offs[i]:offs[i] + int(L)]), tc))
                rec["contact_p_at_L5"] = float(np.mean(precs)) if precs else np.nan
                rec["contact_n_proteins"] = len(precs)

            rows.append(rec)
            print("  L{:<3} ({:>4})  ".format(layer, rec["rel_depth"]) +
                  "  ".join(f"{k}={v:.4f}" for k, v in rec.items()
                            if isinstance(v, float) and np.isfinite(v)))
            del X
        del model

    if not rows:
        raise SystemExit("no results — check --ckpt-root and --arms")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "saefree_by_arm.csv", index=False)

    metrics = [c for c in df.columns if c.startswith("probe_") or c == "contact_p_at_L5"]
    mlm = df[~df.causal].groupby("rel_depth")[metrics].mean()
    clm = df[df.causal].groupby("rel_depth")[metrics].mean()
    common = [d for d in mlm.index if d in clm.index]
    diff = (mlm.loc[common] - clm.loc[common]).reset_index()
    diff.to_csv(out_dir / "saefree_mlm_minus_clm.csv", index=False)

    print("\n" + "=" * 88)
    print("SAE-FREE:  MLM - CLM  (positive = masked better).  No SAE used anywhere.")
    print("=" * 88)
    print(diff.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    for m in metrics:
        if m in diff.columns:
            v = diff[m].to_numpy(float); v = v[np.isfinite(v)]
            if v.size:
                print(f"  {m:<24} mean {v.mean():+.4f}  MLM>CLM at {int((v > 0).sum())}/{v.size}")
    print(f"\nwrote -> {out_dir}/")


if __name__ == "__main__":
    main()
