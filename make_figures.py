#!/usr/bin/env python3
"""
make_figures.py — the paper's figures, regenerated from delivered result CSVs.

Every figure here is drawn from files in a results directory, never from numbers
typed into this script. The one exception is the untrained baseline, which is
labelled as such at its definition below: those six values come from the
2026-08-28 batch and no later delivery has carried the untrained L_struct cells.

USAGE
    python make_figures.py --results <dir> --out paper/figures

Writes PDF (for \\includegraphics) and PNG (to look at) for each figure. Skips
any figure whose data is absent and says which, rather than drawing an empty one.

COLOUR
    Three conditions, categorical slots 1/2/3 of the project palette, validated
    for CVD separation. Series also differ by marker and carry direct labels, so
    identity survives greyscale printing and the aqua's low contrast against
    white.
"""
import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- palette (validated: CVD dE 9.2 deutan, normal 27.6) --------------------
C_NATIVE, C_BLOCK, C_SHUF = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# --- the untrained floor: 2026-08-28 batch, n=3 (seeds 42/43/44). ----------
# Not present as CSVs in any later delivery; see Table 2 of the paper.
UNTRAINED = {("mlm", 11): 0.01821, ("mlm", 14): 0.01778, ("mlm", 18): 0.01765,
             ("clm", 11): 0.01790, ("clm", 14): 0.01846, ("clm", 18): 0.01901}
ARM = {"mlm": "ckpt_mlm_s{}_token", "clm": "ckpt_clm_s{}"}
SEEDS = ("42", "43", "44")
ARMNAME = {"mlm": "Masked", "clm": "Causal"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def mean_delta(p, col="struct_delta"):
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    if not rows or col not in rows[0]:
        return None
    v = [float(r[col]) for r in rows if r[col] not in ("", "nan")]
    return st.mean(v) if v else None


def cells(base, tree, fname="struct_seq_metrics.csv"):
    """(arm, layer) -> path, for the named tree wherever it sits under base."""
    out = {}
    for root in Path(base).rglob(tree):
        if not root.is_dir() or root.name != tree:
            continue
        for p in root.rglob(fname):
            m = re.search(r"layer_(\d+)$", p.parent.name)
            if m:
                out.setdefault((p.parent.parent.name, int(m.group(1))), p)
    return out


def by_depth(cl, arm_key):
    """layer -> (mean over seeds, sd over seeds)."""
    out = {}
    layers = sorted({L for (a, L) in cl if a.startswith(f"ckpt_{arm_key}")})
    for L in layers:
        vals = [mean_delta(cl[(ARM[arm_key].format(s), L)])
                for s in SEEDS if (ARM[arm_key].format(s), L) in cl]
        vals = [v for v in vals if v is not None]
        if vals:
            out[L] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
    return out


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out/name}.pdf and .png")


# ===========================================================================
def fig_depth_profile(base, out):
    nat, shu = cells(base, "outputs_ctrl"), cells(base, "outputs_ctrl_shuf")
    blk = {}
    for t in ("outputs_ctrl_blk16", "outputs_ctrl_blk8"):
        blk.update(cells(base, t))
    if not nat or not shu:
        print("  [skip] depth profile: needs outputs_ctrl and outputs_ctrl_shuf")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    series = ((nat, C_NATIVE, "o", "native"),
              (blk, C_BLOCK, "s", "block-shuffled"),
              (shu, C_SHUF, "^", "order-destroyed"))
    for ax, ak in zip(axes, ("mlm", "clm")):
        lo = min(UNTRAINED[(ak, L)] for L in (11, 14, 18))
        hi = max(UNTRAINED[(ak, L)] for L in (11, 14, 18))
        ax.axhspan(lo, hi, color=MUTED, alpha=0.13, lw=0, zorder=0)
        # park the band label where that panel's curves are not
        lx, lha = (-0.6, "left") if ak == "mlm" else (29.5, "left")
        ax.text(lx, hi, "untrained", ha=lha, va="bottom", fontsize=7, color=MUTED)
        for cl, colour, mk, lab in series:
            d = by_depth(cl, ak)
            if not d:
                continue
            xs = sorted(d)
            ys = [d[x][0] for x in xs]
            es = [d[x][1] for x in xs]
            ax.errorbar(xs, ys, yerr=es, color=colour, marker=mk, ms=4.2, lw=1.6,
                        capsize=2, elinewidth=0.8, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.5, label=lab)
            # Direct-label only on the causal panel, where the three curves stay
            # far apart. On the masked panel native and order-destroyed converge
            # by block 29 and any end-of-line label collides; the legend carries
            # identity there.
            if ak == "clm":
                # block-shuffled ends at block 18 with the native curve running
                # just beneath it, so that one label goes above rather than beside
                dy = 10 if lab == "block-shuffled" else 0
                ax.annotate(lab, (xs[-1], ys[-1]), textcoords="offset points",
                            xytext=(5, dy), fontsize=7, color=colour,
                            weight="bold", va="center")
        ax.set_title(f"{ARMNAME[ak]} arm", loc="left")
        ax.set_xlabel("block")
        ax.set_xticks([0, 4, 7, 11, 14, 18, 22, 26, 29])
        ax.grid(axis="y", zorder=0)
    axes[0].set_xlim(-1.5, 30.5)
    axes[1].set_xlim(-1.5, 46)
    axes[0].set_ylabel(r"mean $L_{\mathrm{struct}}$")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.5,
                   labelspacing=0.3, borderpad=0.2)
    fig.subplots_adjust(wspace=0.06)
    save(fig, out, "fig_depth_profile")


def fig_trained_vs_untrained(base, out):
    nat = cells(base, "outputs_ctrl")
    if not nat:
        print("  [skip] trained-vs-untrained: needs outputs_ctrl")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    lo = min(UNTRAINED.values()); hi = max(UNTRAINED.values())
    ax.axhspan(lo, hi, color=MUTED, alpha=0.13, lw=0, zorder=0)
    ax.text(18.4, hi, "untrained\n(n=3)", ha="left", va="center",
            fontsize=7, color=MUTED)
    for ak, colour, mk in (("mlm", C_NATIVE, "o"), ("clm", C_SHUF, "^")):
        d = by_depth(nat, ak)
        xs = [L for L in (11, 14, 18) if L in d]
        if not xs:
            continue
        ax.errorbar(xs, [d[x][0] for x in xs], yerr=[d[x][1] for x in xs],
                    color=colour, marker=mk, ms=5, lw=1.6, capsize=2.5,
                    elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.5,
                    zorder=3, label=f"{ARMNAME[ak]} trained")
        ax.plot(xs, [UNTRAINED[(ak, x)] for x in xs], marker=mk, ms=5, lw=0,
                mfc="white", mec=colour, mew=1.4, zorder=3)
    ax.set_xticks([11, 14, 18]); ax.set_xlim(9.5, 22)
    ax.set_xlabel("block"); ax.set_ylabel(r"mean $L_{\mathrm{struct}}$")
    ax.grid(axis="y", zorder=0)
    ax.legend(frameon=False, loc="center left", handlelength=1.4)
    # the band between the arms is empty; the foot of the axes is not
    ax.text(0.03, 0.30, "filled = trained\nopen = untrained", transform=ax.transAxes,
            fontsize=7, color=MUTED, va="top")
    save(fig, out, "fig_trained_vs_untrained")


def fig_denominator(base, out):
    f = "struct_seq_metrics_denominators.csv"
    nat, shu = cells(base, "outputs_ctrl", f), cells(base, "outputs_ctrl_shuf", f)
    ks = sorted(set(nat) & set(shu))
    if not ks:
        print("  [skip] denominator: needs struct_seq_metrics_denominators.csv")
        return
    modes = ("sd", "fixed", "iqr", "rank")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
    for ax, m in zip(axes, modes):
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        n_up = 0
        # the two visible clusters are the two arms, so encode that by marker
        # rather than leaving the reader to guess
        for ak, mk in (("mlm", "o"), ("clm", "^")):
            ks_a = [k for k in ks if k[0].startswith(f"ckpt_{ak}")]
            rises = [mean_delta(shu[k], f"struct_delta_{m}") -
                     mean_delta(nat[k], f"struct_delta_{m}") for k in ks_a]
            n_up += sum(1 for r in rises if r > 0)
            xs = [0.5 + (i - len(rises) / 2) * 0.006 for i in range(len(rises))]
            ax.scatter(xs, rises, s=16, marker=mk, color=C_SHUF, zorder=3,
                       edgecolor="white", linewidth=0.5,
                       label={"mlm": "masked", "clm": "causal"}[ak])
        # the count goes in the title: at the foot of the panel it collides
        # with the lowest masked points
        ax.set_title(f"{m}   {n_up}/{len(ks)}", loc="center", color=INK)
        ax.set_xticks([]); ax.set_xlim(0.44, 0.56)
        ax.grid(axis="y", zorder=0)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    axes[0].set_ylabel(r"$L_{\mathrm{struct}}$ rise" "\n" r"(order-destroyed $-$ native)")
    axes[0].legend(frameon=False, loc="center left", fontsize=7,
                   handletextpad=0.2, borderpad=0.1)
    fig.suptitle("every cell rises under every denominator", x=0.02, ha="left",
                 fontsize=9.5, y=1.02)
    fig.subplots_adjust(wspace=0.45)
    save(fig, out, "fig_denominator")


def fig_bh(base, out):
    log = next((p for p in Path(base).rglob("s5_bh_*.log")), None)
    if not log:
        print("  [skip] BH: no s5_bh_*.log in the delivery")
        return
    rows, omni = [], None
    for line in log.read_text().splitlines():
        m = re.match(r"\s*(\d+)%\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
                     r"[\d.]+\s+[\d.]+\s+[\d.]+\s+(True|False)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                         float(m.group(4)), m.group(5) == "True"))
        if "[full, n=" in line:
            mm = re.search(r"mean d = ([+\-\d.]+)\s+95% CI \[([+\-\d.]+), ([+\-\d.]+)\]", line)
            if mm:
                omni = tuple(float(x) for x in mm.groups())
    if not rows:
        print("  [skip] BH: could not parse the log")
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    if omni:
        ax.axhspan(omni[1], omni[2], color=C_NATIVE, alpha=0.12, lw=0, zorder=0)
        ax.axhline(omni[0], color=C_NATIVE, lw=1.1, ls="--", zorder=1)
        # left end: the shallow depths sit near zero, so the band label is clear
        # there, whereas the 100% CI runs right through the right-hand side
        ax.text(-6, omni[0], f"omnibus {omni[0]:+.3f}\n95% CI [{omni[1]:+.3f}, {omni[2]:+.3f}]",
                va="bottom", ha="left", fontsize=6.5, color=C_NATIVE)
    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
    # CI drawn as a range and the point drawn separately, rather than as an
    # errorbar: at 25% depth the bootstrap point estimate lies ABOVE its own
    # upper bound, which errorbar cannot express and would silently hide.
    outside = []
    for d, pt, lo, hi, sig in rows:
        col = C_SHUF if sig else MUTED
        ax.vlines(d, lo, hi, color=col, lw=1.2, zorder=2)
        ax.hlines([lo, hi], d - 1.6, d + 1.6, color=col, lw=1.0, zorder=2)
        ax.plot([d], [pt], marker="o", ms=5, lw=0, zorder=3,
                mfc=col if sig else "white", mec=col, mew=1.2)
        if not (lo <= pt <= hi):
            outside.append(d)
            ax.annotate("", xy=(d, pt), xytext=(d + 9, pt + 0.075),
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED))
            ax.text(d + 9.5, pt + 0.078, "point outside\nits own CI", fontsize=6.5,
                    color=MUTED, va="center")
    if outside:
        print(f"  NOTE: point estimate outside its CI at depth(s) {outside}% "
              "— flagged on the figure")
    ax.set_xlabel("relative depth"); ax.set_ylabel(r"between-arm $d$")
    ax.set_xticks([r[0] for r in rows])
    ax.set_xticklabels([f"{r[0]}%" for r in rows], fontsize=7)
    ax.grid(axis="y", zorder=0); ax.set_xlim(-9, 116)
    n_sig = sum(1 for r in rows if r[4])
    ax.legend(handles=[
        Line2D([], [], marker="o", lw=0, mfc=C_SHUF, mec=C_SHUF, ms=5,
               label=f"survives BH ({n_sig}/{len(rows)})"),
        Line2D([], [], marker="o", lw=0, mfc="white", mec=MUTED, ms=5, label="does not"),
    ], frameon=False, loc="lower right", handlelength=1.0)
    save(fig, out, "fig_bh_depths")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    base, out = Path(a.results), Path(a.out)
    if not base.is_dir():
        print(f"ERROR: no such directory: {base}", file=sys.stderr)
        return 2
    print(f"figures from {base} -> {out}")
    for fn in (fig_depth_profile, fig_trained_vs_untrained, fig_denominator, fig_bh):
        fn(base, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
