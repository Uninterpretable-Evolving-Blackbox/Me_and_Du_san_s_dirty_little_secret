#!/usr/bin/env python3
"""
analyze_directional.py — the mechanism test: does the MLM advantage live where
bidirectionality actually buys you something?

THE LOGIC
  A causal model's representation at residue i cannot depend on any token after i.
  So for a contact (i, j):
      j > i  (downstream) — information the CLM structurally CANNOT have at i
      j < i  (upstream)   — information available to BOTH arms
  If the masked model's advantage on structural locality comes from bidirectional
  attention, it must be LARGER on downstream contacts.

THE STATISTIC — an interaction, not a raw difference
      Delta_down = d(MLM, CLM | downstream contacts)
      Delta_up   = d(MLM, CLM | upstream contacts)
      INTERACTION = Delta_down - Delta_up          <- the prediction is > 0

  Why the interaction and not the raw up/down numbers: `_cohens_d_vectorized` takes
  its global mean over all residues including degree-0 ones, and splitting a graph by
  direction leaves many more residues with degree 0 (early residues have few upstream
  partners, late residues few downstream). That makes up and down incommensurable
  WITHIN a model. It cancels in the interaction because the contact graph is a property
  of the PROTEIN and is byte-identical across the two arms being compared.

THE CONTROL — run in the same pass, and it is what makes the test falsifiable
  The identical split is applied to the SEQUENTIAL graph (-2,-1 vs +1,+2). A generic
  "the masked model is better at anything forward-looking" effect produces a positive
  interaction in BOTH graphs. A specifically STRUCTURAL mechanism produces it only in
  the contact graph. So the quantity to report is:

      structural interaction  -  sequential interaction

  A symmetric MLM advantage (interaction ~ 0) REFUTES the bidirectionality explanation.
  That is a real, publishable outcome, not a failed experiment — it would say the
  effect comes from something other than access to downstream context.

Input: `struct_seq_directional.csv` written by `cpu_stage.py --split-direction updown`.

Usage
  .venv/bin/python analyze_directional.py \
      --root outputs_ctrl \
      --mlm ckpt_mlm_s42_token,ckpt_mlm_s43_token,ckpt_mlm_s44_token \
      --clm ckpt_clm_s42,ckpt_clm_s43,ckpt_clm_s44 \
      --depths 11,14,18
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

DEPTH_LABELS = {0: "0%", 4: "13%", 7: "25%", 11: "38%", 14: "50%",
                18: "63%", 22: "75%", 26: "88%", 29: "100%"}


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def load_cell(root, name, layer):
    p = Path(root) / name / f"layer_{layer}" / "struct_seq_directional.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_ctrl")
    ap.add_argument("--mlm", required=True, help="comma-separated MLM cell names, one per seed")
    ap.add_argument("--clm", required=True, help="comma-separated CLM cell names, seed-aligned")
    ap.add_argument("--depths", default="11,14,18",
                    help="block indices; default 38/50/63%% where the effect is largest")
    ap.add_argument("--out", default="results_rigor/directional_mechanism.csv")
    args = ap.parse_args()

    mlm_cells = [c.strip() for c in args.mlm.split(",") if c.strip()]
    clm_cells = [c.strip() for c in args.clm.split(",") if c.strip()]
    if len(mlm_cells) != len(clm_cells):
        raise SystemExit(f"--mlm has {len(mlm_cells)} cells but --clm has {len(clm_cells)}; "
                         "they must be seed-aligned")
    depths = [int(d) for d in args.depths.split(",") if d.strip()]

    rows = []
    for layer in depths:
        for si, (mc, cc) in enumerate(zip(mlm_cells, clm_cells)):
            m = load_cell(args.root, mc, layer)
            c = load_cell(args.root, cc, layer)
            if m is None or c is None:
                print(f"  [skip] layer {layer} seed-slot {si}: missing directional CSV "
                      f"({'MLM ' if m is None else ''}{'CLM' if c is None else ''})")
                continue
            rec = dict(layer=layer, rel_depth=DEPTH_LABELS.get(layer, f"L{layer}"),
                       mlm_cell=mc, clm_cell=cc)
            for graph in ("struct", "seq"):
                up = cohens_d(m[f"{graph}_up_delta"], c[f"{graph}_up_delta"])
                dn = cohens_d(m[f"{graph}_down_delta"], c[f"{graph}_down_delta"])
                rec[f"{graph}_up"] = up
                rec[f"{graph}_down"] = dn
                rec[f"{graph}_interaction"] = dn - up
            # combined column is present in the same file — a free consistency check
            if "struct_delta" in m.columns and "struct_delta" in c.columns:
                rec["struct_combined"] = cohens_d(m["struct_delta"], c["struct_delta"])
            rec["net_interaction"] = rec["struct_interaction"] - rec["seq_interaction"]
            rows.append(rec)

    if not rows:
        raise SystemExit("no cells found — did you run cpu_stage.py --split-direction updown?")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("=" * 100)
    print("MECHANISM TEST — d(MLM, CLM) split by contact direction.  Prediction: down > up")
    print("=" * 100)
    show = ["rel_depth", "struct_up", "struct_down", "struct_interaction",
            "seq_up", "seq_down", "seq_interaction", "net_interaction"]
    print(df[[c for c in show if c in df.columns]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n" + "-" * 100)
    n_seeds = len(mlm_cells)
    for col, label in (("struct_interaction", "STRUCTURAL interaction (down - up)"),
                       ("seq_interaction",    "sequential interaction  (control)"),
                       ("net_interaction",    "NET  (structural - sequential)")):
        v = df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        # Unit of replication is the training run, so aggregate to one value per seed
        # before testing — depths within a seed are not independent.
        per_seed = (df.groupby("mlm_cell")[col].mean().to_numpy(dtype=float)
                    if n_seeds > 1 else v)
        per_seed = per_seed[np.isfinite(per_seed)]
        line = f"  {label:<38} mean {v.mean():+.4f}   positive at {int((v > 0).sum())}/{v.size} cells"
        if per_seed.size >= 2:
            se = per_seed.std(ddof=1) / np.sqrt(per_seed.size)
            if se > 0:
                t = per_seed.mean() / se
                p = 2 * tdist.sf(abs(t), df=per_seed.size - 1)
                line += f"   | across seeds: t={t:+.2f}, p={p:.3f} (df={per_seed.size-1})"
        print(line)

    net = df["net_interaction"].to_numpy(dtype=float)
    net = net[np.isfinite(net)]
    print("\n  READING THIS:")
    if net.size and net.mean() > 0 and (net > 0).mean() >= 0.75:
        print("    Net interaction is positive and consistent. The masked model's advantage is")
        print("    concentrated on contacts a causal model structurally cannot see, and this is")
        print("    NOT explained by a generic forward-looking advantage (the sequential control).")
        print("    That is mechanistic evidence tied directly to the manipulated variable.")
    elif net.size and abs(net.mean()) < 0.05:
        print("    Net interaction is ~0. The MLM advantage is SYMMETRIC in contact direction,")
        print("    which REFUTES the bidirectional-access explanation. Report it as such — the")
        print("    effect is real but its mechanism is something other than downstream context.")
    else:
        print("    Mixed or negative. Do not force an interpretation; report the table and say")
        print("    the mechanism test was inconclusive at this depth set.")
    print(f"\n  wrote -> {args.out}")
    print("  CAVEAT: never report struct_up vs struct_down within a single arm as a structural")
    print("          fact — the two are on different scales by degree. Only the interaction is")
    print("          interpretable (see cpu_stage.py's directional-split header).")


if __name__ == "__main__":
    main()
