#!/usr/bin/env python3
"""
summarize_reviewer_batch.py — turn the stage outputs into the comparisons that
actually answer the reviewers. Run this after the batch; send me its output.

WHY THIS EXISTS
---------------
Stages 7, 9, 10, 14, 15 and 16 each write per-cell files and then print a line
telling a human to do the comparison. That is where the errors happen. Two
aggregation mistakes have already been made on this project's own data: a pivot
that silently paired shuffled seed 42 against real seeds 43/44, and a concept-F1
figure that was a max over depths being compared against a per-depth mean. Both
looked plausible and both were wrong.

So every comparison this batch exists to make is computed here, once, in code, with
the pairing written down explicitly.

USAGE
  python summarize_reviewer_batch.py                     # everything it can find
  python summarize_reviewer_batch.py --only stage14
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS_DEFAULT = [11, 14, 18]


def _mean_sd(path):
    try:
        return float(pd.read_csv(path).struct_delta.mean())
    except Exception:
        return np.nan


def stage7(root=".", arms=("ckpt_mlm_s42_token", "ckpt_clm_s42"), layers=LAYERS_DEFAULT):
    """Dictionary-seed variance vs the model-seed variance the paper reports (SD 0.0234)."""
    print("\n=== STAGE 7 — dictionary seed vs model seed ===")
    rows = []
    for arm in arms:
        for L in layers:
            vals = {}
            for tag, tree in [("d42", "outputs_ctrl"), ("d43", "outputs_ctrl_dseed43"),
                              ("d44", "outputs_ctrl_dseed44")]:
                f = Path(root) / tree / arm / f"layer_{L}" / "struct_seq_metrics.csv"
                if f.exists():
                    vals[tag] = _mean_sd(f)
            if len(vals) >= 2:
                v = np.array(list(vals.values()))
                rows.append(dict(arm=arm, layer=L, n_dict_seeds=len(v),
                                 mean=v.mean(), sd_across_dict_seeds=v.std(ddof=1)))
    if not rows:
        print("  (no dseed trees yet)"); return
    df = pd.DataFrame(rows)
    print(df.round(5).to_string(index=False))
    print(f"\n  median SD across DICTIONARY seeds : {df.sd_across_dict_seeds.median():.5f}")
    print( "  SD across MODEL seeds (paper S3)  : 0.02340")
    print("  If the first is comparable to or larger than the second, S3's seed-level")
    print("  intervals are understated and the omnibus p = 0.0028 is optimistic.")


def stage9(root="."):
    """Is the 5-permutation null undersampled?"""
    print("\n=== STAGE 9 — permutation count, 5 vs 25 ===")
    a = Path(root) / "results_nshuffle_sensitivity/nshuf5/contact_def_sweep.csv"
    b = Path(root) / "results_nshuffle_sensitivity/nshuf25/contact_def_sweep.csv"
    if not (a.exists() and b.exists()):
        print("  (not run)"); return
    A, B = pd.read_csv(a), pd.read_csv(b)
    key = [c for c in ("layer", "cutoff_A", "seq_gap") if c in A.columns]
    m = A.merge(B, on=key, suffixes=("_n5", "_n25"))
    m["abs_change"] = (m.d_n25 - m.d_n5).abs()
    # NB "pct_change" as an attribute collides with DataFrame.pct_change, so these
    # are read with [] rather than dot access. That collision silently turned the
    # summary line into an AttributeError on the first real run.
    m["pct_shift"] = 100 * m["abs_change"] / m.d_n5.abs().clip(lower=1e-9)
    print(m[key + ["d_n5", "d_n25", "abs_change", "pct_shift"]].round(4).to_string(index=False))
    print(f"\n  median |change| {m['abs_change'].median():.4f}  "
          f"({m['pct_shift'].median():.1f}% of the effect)   "
          f"max {m['abs_change'].max():.4f}   sign changes {(m.d_n5*m.d_n25 < 0).sum()}/{len(m)}")
    print("  A material move means five draws do not estimate the null tightly enough.")


def stage10(root="."):
    """L_struct with no autoencoder: does the shuffled/real gap survive?"""
    print("\n=== STAGE 10 — raw dimensions (no SAE) + perplexity ===")
    d = Path(root) / "results_raw_coactivation"
    if not d.exists():
        print("  (not run)"); return
    df = pd.concat([pd.read_csv(f).assign(file=f.name) for f in d.glob("*.csv")],
                   ignore_index=True)
    df["tree"] = np.where(df.file.str.startswith("shuf"), "shuffled", "real")
    df["arm"] = np.where(df.file.str.contains("mlm"), "masked", "causal")
    p = df.pivot_table(index=["arm", "layer"], columns="tree", values="mean_struct_delta")
    if {"real", "shuffled"} <= set(p.columns):
        p["shuf_minus_real"] = p["shuffled"] - p["real"]
        p["ratio"] = p["shuffled"] / p["real"].replace(0, np.nan)
    print(p.round(5).to_string())
    print("\n  THE QUESTION: is shuffled still above real with NO dictionary in the path?")
    print("  yes -> no instrumental account of S4.3 can hold, the effect is in the model")
    print("  no  -> the autoencoder is the story and S4.3 resolves to one cause")
    ppl = df.dropna(subset=["perplexity"])[["file", "perplexity", "ppl_kind"]].drop_duplicates()
    if len(ppl):
        print("\n  perplexity (causal and pseudo-masked are DIFFERENT quantities, never compare):")
        print(ppl.round(3).to_string(index=False))


def stage11(root="."):
    """Pairwise contact probes, and the positive control that makes them readable."""
    print("\n=== STAGE 11 — pairwise contact probes ===")
    d = Path(root) / "results_pairwise_probe"
    if not d.exists():
        print("  (not run)"); return
    rows = []
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        sm = j.get("separation_matched", {})
        sh = j.get("separation_matched_shuffled_labels", {})
        rows.append(dict(cell=f.stem, mode=j.get("mode"), auroc=sm.get("auroc"),
                         shuffled_label_control=sh.get("auroc")))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    pc = df[df.cell.str.contains("POSITIVE_CONTROL")]
    if len(pc) and pd.notna(pc.auroc.iloc[0]):
        v = float(pc.auroc.iloc[0])
        verdict = ("PROBE HAS POWER — nulls elsewhere are meaningful" if v > 0.56
                   else "PROBE TOO WEAK — every other number here is uninterpretable")
        print(f"\n  positive control (ESM-2 raw): {v:.4f}   -> {verdict}")
    else:
        print("\n  !! NO POSITIVE CONTROL FOUND. Without it a null here cannot be")
        print("     distinguished from a probe too weak to detect anything.")
    if df.shuffled_label_control.notna().any():
        w = df[df.shuffled_label_control.sub(0.5).abs() > 0.06]
        if len(w):
            print("\n  !! shuffled-label control away from 0.5 in these cells — possible leakage:")
            print(w.round(4).to_string(index=False))


def stage14(root="."):
    """Simon & Zou's own metric: does IT inflate on shuffled models?"""
    print("\n=== STAGE 14 — Simon & Zou's metric, exactly as published ===")
    d = Path(root) / "results_interplm_metric"
    if not d.exists():
        print("  (not run)"); return
    rows = []
    for f in sorted(d.glob("*.csv")):
        df = pd.read_csv(f).dropna(subset=["d_struct"])
        if not len(df):
            continue
        bonf = 0.05 / len(df)
        rows.append(dict(cell=f.stem,
                         tree="shuffled" if "shuf" in f.stem else "real",
                         arm="masked" if "mlm" in f.stem else "causal",
                         n_scored=len(df), mean_d_struct=df.d_struct.mean(),
                         n_significant=int((df.p_struct < bonf).sum()),
                         median_ratio=df[df.p_struct < bonf].ratio.median()))
    if not rows:
        print("  (no scored features)"); return
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    p = df.pivot_table(index="arm", columns="tree",
                       values=["mean_d_struct", "n_significant"])
    print("\n" + p.round(4).to_string())
    print("\n  THE PAPER TURNS ON THIS. If shuffled > real on mean_d_struct or on the")
    print("  count of significant structural features, the failure is NOT specific to")
    print("  our variant and the cautionary tale has an audience. If not, our anchor")
    print("  and gate are what break, and the paper narrows to a claim about them.")


def stage15(root="."):
    """The matched untrained null, for all three claims."""
    print("\n=== STAGE 15 — matched untrained baseline ===")
    any_found = False
    for arm in ("ckpt_mlm_s42_token", "ckpt_clm_s42"):
        for L in LAYERS_DEFAULT:
            t = Path(root) / "outputs_ctrl" / arm / f"layer_{L}" / "struct_seq_metrics.csv"
            r = Path(root) / "outputs_ctrl_randominit" / arm / f"layer_{L}" / "struct_seq_metrics.csv"
            if t.exists() and r.exists():
                any_found = True
                print(f"  L_struct {arm:22} L{L:<3} trained {_mean_sd(t):+.5f}   "
                      f"untrained {_mean_sd(r):+.5f}")
    a = Path(root) / "results_ctrl_saefree/saefree_by_arm.csv"
    b = Path(root) / "results_ctrl_saefree_randominit/saefree_by_arm.csv"
    if a.exists() and b.exists():
        any_found = True
        A, B = pd.read_csv(a), pd.read_csv(b)
        print("\n  probe AUROC (0.5 = chance), trained vs untrained:")
        for c in [x for x in A.columns if "auroc" in x]:
            print(f"    {c:24} trained {A[c].min():.3f}-{A[c].max():.3f}   "
                  f"untrained {B[c].min():.3f}-{B[c].max():.3f}")
    cf_t = sorted((Path(root) / "results_concept_f1").glob("*/layer_*/concept_f1.csv"))
    cf_r = sorted((Path(root) / "results_concept_f1_randominit").glob("*/layer_*/concept_f1.csv"))
    if cf_t and cf_r:
        any_found = True
        def best(fs):
            d = pd.concat([pd.read_csv(f) for f in fs])
            return d.groupby("concept").test_f1.max()
        T, R = best(cf_t), best(cf_r)
        j = pd.concat([T.rename("trained"), R.rename("untrained")], axis=1).dropna()
        j["gap"] = j.trained - j.untrained
        print("\n  concept-F1, best feature per concept, SAME aggregation both sides:")
        print(j.sort_values("gap", ascending=False).round(3).to_string())
        print("\n  Concepts with a small gap are NOT evidence of learned interpretable")
        print("  features. This is the comparison the ESM-2 baseline could not make.")
    if not any_found:
        print("  (not run)")


def stage16(root="."):
    """Table 1 at three seeds instead of one."""
    print("\n=== STAGE 16 — shuffled control at three seeds ===")
    rows = []
    for s in (42, 43, 44):
        for obj, arm_r in [("mlm", "ckpt_mlm_s{}_token"), ("clm", "ckpt_clm_s{}")]:
            arm = arm_r.format(s)
            for L in LAYERS_DEFAULT:
                real = Path(root) / "outputs_ctrl" / arm / f"layer_{L}" / "struct_seq_metrics.csv"
                shuf = Path(root) / "outputs_ctrl_shuf" / arm / f"layer_{L}" / "struct_seq_metrics.csv"
                if real.exists() and shuf.exists():
                    r, sh = _mean_sd(real), _mean_sd(shuf)
                    rows.append(dict(seed=s, arm="masked" if obj == "mlm" else "causal",
                                     layer=L, real=r, shuffled=sh,
                                     ratio=sh / r if r else np.nan))
    if not rows:
        print("  (seeds 43/44 not evaluated yet — stage 5 trains, stage 16 evaluates)")
        return
    df = pd.DataFrame(rows)
    print(df.round(5).to_string(index=False))
    g = df.groupby(["arm", "layer"]).ratio.agg(["mean", "std", "count"])
    print("\n  ratio across seeds:")
    print(g.round(4).to_string())
    print(f"\n  cells with ratio > 1: {(df.ratio > 1).sum()} / {len(df)}")
    n_seeds = df.seed.nunique()
    if n_seeds < 2:
        print(f"\n  !! ONLY SEED {df.seed.iloc[0]} PRESENT. Stage 5 trains the extra seeds and")
        print("     stage 16 evaluates them; neither has run, so this is still n=1 per arm.")
    else:
        print(f"\n  {n_seeds} seeds present -- the invalidity result is no longer n=1 per arm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    fns = dict(stage7=stage7, stage9=stage9, stage10=stage10, stage11=stage11,
               stage14=stage14, stage15=stage15, stage16=stage16)
    for name, fn in fns.items():
        if args.only and args.only != name:
            continue
        try:
            fn(args.root)
        except Exception as e:
            print(f"\n=== {name.upper()} — FAILED: {type(e).__name__}: {e} ===")


if __name__ == "__main__":
    main()
