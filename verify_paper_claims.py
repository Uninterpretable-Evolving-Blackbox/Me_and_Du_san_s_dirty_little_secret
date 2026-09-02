#!/usr/bin/env python3
"""
verify_paper_claims.py — recompute the paper's numbers from the delivered archives.

WHY THIS EXISTS
---------------
Every number in the paper was read off a batch by hand at some point. Two
aggregation mistakes have already been made on this project's own data (a pivot
that paired shuffled seed 42 against real seeds 43/44, and a concept-F1 figure
that was a max over depths compared against a per-depth mean). Both looked
plausible. This file is the answer to "can anyone re-derive Table 2 from the
CSVs?" — it states each claim, recomputes it, and prints the two side by side.

It reads only. No GPU, no checkpoints, no network, seconds to run.

USAGE
    python verify_paper_claims.py --results <dir>

<dir> is anywhere the archives were unpacked, or a directory of .tgz files —
they are extracted to a temporary directory and left alone. Sub-directories are
discovered by name, so an incomplete delivery reports MISSING per claim rather
than failing as a whole.

EXIT STATUS
    0  every claim the data covers matches the paper
    1  at least one claim CHANGED — the paper says something the data does not
    2  the results directory could not be read at all

A CHANGED verdict is not a bug report. These runs were expected to move some
numbers; the point is that the move is visible and attributable rather than
discovered by a reviewer.
"""
import argparse
import csv
import glob
import math
import re
import statistics as st
import sys
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict

ARMS = {"mlm": "ckpt_mlm_s{}_token", "clm": "ckpt_clm_s{}"}
SEEDS = ("42", "43", "44")
CTRL_DEPTHS = (11, 14, 18)
TOL = 5e-5            # absolute tolerance on a mean L_struct quoted to 5 dp

results = []          # (section, claim, paper, computed, verdict, note)


def record(section, claim, paper, computed, verdict, note=""):
    results.append((section, claim, paper, computed, verdict, note))


def mean_col(path, col="struct_delta"):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows or col not in rows[0]:
        return None
    vals = [float(r[col]) for r in rows if r[col] not in ("", "nan")]
    return st.mean(vals) if vals else None


def find_cells(root, fname="struct_seq_metrics.csv"):
    """(arm, layer) -> path, for every cell under root."""
    out = {}
    if not root or not Path(root).is_dir():
        return out
    for p in Path(root).rglob(fname):
        m = re.search(r"layer_(\d+)", str(p.parent.name))
        if m:
            out[(p.parent.parent.name, int(m.group(1)))] = p
    return out


def discover(base):
    """Locate each tree by name, wherever it sits under base."""
    d = {}
    for key, pat in (("native", "outputs_ctrl"), ("shuf", "outputs_ctrl_shuf"),
                     ("blk", "outputs_ctrl_blk*"), ("interplm", "results_interplm_metric"),
                     ("probe18", "results_ctrl_saefree_L18"),
                     ("probe18mlp", "results_ctrl_saefree_L18_mlp")):
        hits = [p for p in Path(base).rglob(pat) if p.is_dir()]
        # outputs_ctrl also matches outputs_ctrl_shuf/_blk under rglob's glob rules
        if key == "native":
            hits = [p for p in hits if p.name == "outputs_ctrl"]
        d[key] = hits
    return d


def merged(paths, fname="struct_seq_metrics.csv"):
    """Union of cells across archives; a cell present twice must agree."""
    out, clashes = {}, []
    for root in paths:
        for k, v in find_cells(root, fname).items():
            if k in out:
                a, b = mean_col(out[k]), mean_col(v)
                if a is not None and b is not None and abs(a - b) > 1e-9:
                    clashes.append((k, a, b))
            else:
                out[k] = v
    return out, clashes


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    base = Path(args.results)
    if not base.exists():
        print(f"ERROR: no such directory: {base}", file=sys.stderr)
        return 2

    tmp = None
    tgz = sorted(base.rglob("*.tgz"))
    if tgz and not any(base.rglob("struct_seq_metrics.csv")):
        tmp = tempfile.mkdtemp(prefix="verify_")
        for t in tgz:
            with tarfile.open(t) as tf:
                tf.extractall(tmp)
        base = Path(tmp)
        print(f"unpacked {len(tgz)} archive(s) to {tmp}\n")

    tr = discover(base)
    nat, nat_clash = merged(tr["native"])
    shu, shu_clash = merged(tr["shuf"])
    blk, _ = merged(tr["blk"])

    print("=" * 92)
    print(f"verify_paper_claims.py   {base}")
    print(f"  native cells {len(nat)} | order-destroyed {len(shu)} | block-shuffled {len(blk)}")
    print("=" * 92)

    # ---- 0. cross-archive consistency -------------------------------------
    for name, cl in (("native", nat_clash), ("order-destroyed", shu_clash)):
        record("consistency", f"{name} cells agree across archives",
               "identical", f"{len(cl)} clash(es)",
               "PASS" if not cl else "CHANGED",
               "" if not cl else "; ".join(f"{k} {a:.6f} vs {b:.6f}" for k, a, b in cl[:3]))

    # ---- 1. §4.1 corpus control ------------------------------------------
    ctrl = [k for k in sorted(set(nat) & set(shu)) if k[1] in CTRL_DEPTHS]
    if ctrl:
        up = sum(1 for k in ctrl if mean_col(shu[k]) > mean_col(nat[k]))
        record("4.1", "corpus control at blocks 11/14/18", "18/18",
               f"{up}/{len(ctrl)}", "PASS" if (up, len(ctrl)) == (18, 18) else "CHANGED")
    else:
        record("4.1", "corpus control at blocks 11/14/18", "18/18", "-", "MISSING")

    allk = sorted(set(nat) & set(shu))
    if allk:
        up = sum(1 for k in allk if mean_col(shu[k]) > mean_col(nat[k]))
        falls = [f"{a} L{l}" for (a, l) in allk if mean_col(shu[(a, l)]) <= mean_col(nat[(a, l)])]
        record("4.1", "corpus control, all depths delivered", "(new)",
               f"{up}/{len(allk)}", "INFO", ("falls: " + ", ".join(falls)) if falls else "")

    # ---- 2. §4.1 the four quoted means ------------------------------------
    for arm_key, tree, paper in (("clm", nat, 0.00240), ("mlm", nat, 0.01655),
                                 ("clm", shu, 0.05201), ("mlm", shu, 0.02730)):
        cells = [tree[(ARMS[arm_key].format(s), L)]
                 for s in SEEDS for L in CTRL_DEPTHS
                 if (ARMS[arm_key].format(s), L) in tree]
        cond = "native" if tree is nat else "order-destroyed"
        if len(cells) == 9:
            got = st.mean(mean_col(c) for c in cells)
            record("4.1", f"{arm_key.upper()} {cond} mean over 11/14/18",
                   f"{paper:+.5f}", f"{got:+.5f}",
                   "PASS" if abs(got - paper) <= args.tol else "CHANGED")
        else:
            record("4.1", f"{arm_key.upper()} {cond} mean over 11/14/18",
                   f"{paper:+.5f}", f"{len(cells)}/9 cells", "MISSING")

    # ---- 3. Table 2 trained column + Delta --------------------------------
    TAB2 = {("mlm", 11): 0.01881, ("mlm", 14): 0.01667, ("mlm", 18): 0.01417,
            ("clm", 11): 0.00162, ("clm", 14): 0.00144, ("clm", 18): 0.00413}
    deltas, ok = [], True
    for (ak, L), paper in TAB2.items():
        cs = [nat[(ARMS[ak].format(s), L)] for s in SEEDS if (ARMS[ak].format(s), L) in nat]
        if len(cs) == 3:
            got = st.mean(mean_col(c) for c in cs)
            v = "PASS" if abs(got - paper) <= args.tol else "CHANGED"
            ok &= v == "PASS"
            record("Table 2", f"trained {ak.upper()} block {L}", f"{paper:+.5f}",
                   f"{got:+.5f}", v)
        else:
            record("Table 2", f"trained {ak.upper()} block {L}", f"{paper:+.5f}",
                   f"{len(cs)}/3 seeds", "MISSING"); ok = False
    if ok:
        for L in CTRL_DEPTHS:
            m = st.mean(mean_col(nat[(ARMS['mlm'].format(s), L)]) for s in SEEDS)
            c = st.mean(mean_col(nat[(ARMS['clm'].format(s), L)]) for s in SEEDS)
            deltas.append(m - c)
        got = st.mean(deltas)
        record("2.2", "Delta, original split, mean over 11/14/18", "+0.01415",
               f"{got:+.5f}", "PASS" if abs(got - 0.01415) <= args.tol else "CHANGED")

    # ---- 4. denominator ---------------------------------------------------
    dn, _ = merged(tr["native"], "struct_seq_metrics_denominators.csv")
    ds, _ = merged(tr["shuf"], "struct_seq_metrics_denominators.csv")
    ks = sorted(set(dn) & set(ds))
    if ks:
        for mode in ("sd", "fixed", "iqr", "rank"):
            r = [mean_col(ds[k], f"struct_delta_{mode}") - mean_col(dn[k], f"struct_delta_{mode}")
                 for k in ks if mean_col(ds[k], f"struct_delta_{mode}") is not None]
            if r:
                record("5", f"rise survives the {mode} denominator", "not tested",
                       f"{sum(1 for x in r if x > 0)}/{len(r)} rise, median {st.median(r):+.5f}",
                       "PASS" if all(x > 0 for x in r) else "CHANGED")
    else:
        record("5", "denominator variants", "not tested", "-", "MISSING")

    # ---- 5. block shuffle --------------------------------------------------
    bk = sorted(set(nat) & set(blk))
    if bk:
        per = defaultdict(list)
        for k in bk:
            per["mlm" if "mlm" in k[0] else "clm"].append(mean_col(blk[k]) - mean_col(nat[k]))
        tot = sum(len(v) for v in per.values())
        up = sum(1 for v in per.values() for x in v if x > 0)
        record("5", "second destruction procedure (block shuffle)", "not run",
               f"{up}/{tot} rise", "INFO",
               " | ".join(f"{a.upper()} {sum(1 for x in v if x>0)}/{len(v)}"
                          for a, v in sorted(per.items())))
    else:
        record("5", "second destruction procedure (block shuffle)", "not run", "-", "MISSING")

    # ---- 6. d_struct -------------------------------------------------------
    ip = [p for d in tr["interplm"] for p in Path(d).glob("*.csv")]
    if ip:
        b = defaultdict(list)
        for p in ip:
            m = re.match(r"(outputs_ctrl(?:_shuf)?|randominit_s\d+)_ckpt_(mlm|clm)"
                         r"_s\d+(?:_token)?_L\d+_gate-(\w+)\.csv", p.name)
            if not m:
                continue
            cond, arm, gate = m.groups()
            cond = "randominit" if cond.startswith("randominit") else cond
            rows = list(csv.DictReader(open(p)))
            col = next((c for c in rows[0] if "d_struct" in c or c == "d"), None) if rows else None
            if col:
                v = [float(r[col]) for r in rows if r[col] not in ("", "nan")]
                if v:
                    b[(gate, arm, cond)].append(st.mean(v))
        for gate in ("global", "raw"):
            for arm in ("clm", "mlm"):
                n = b.get((gate, arm, "outputs_ctrl"), [])
                s = b.get((gate, arm, "outputs_ctrl_shuf"), [])
                if n and s:
                    up = sum(1 for x, y in zip(sorted(n), sorted(s)) if y > x)
                    exp = "fails" if arm == "clm" else "passes"
                    got = "fails" if st.mean(s) > st.mean(n) else "passes"
                    record("4.6", f"d_struct corpus control, {arm.upper()}, gate={gate}",
                           exp, f"{got} ({up}/{len(n)} cells)",
                           "PASS" if got == exp else "CHANGED")
                u = b.get((gate, arm, "randominit"), [])
                if n and u:
                    record("Table 5", f"d_struct random-init, {arm.upper()}, gate={gate}",
                           "no verdict - not run",
                           f"{'fails' if st.mean(u) >= st.mean(n) else 'passes'} "
                           f"(trained {st.mean(n):+.4f} vs untrained {st.mean(u):+.4f})", "INFO")
    else:
        record("4.6", "d_struct cells", "-", "-", "MISSING")

    # ---- 7. probes ---------------------------------------------------------
    for tag, dirs in (("linear", tr["probe18"]), ("MLP", tr["probe18mlp"])):
        f = next((Path(d) / "saefree_by_arm.csv" for d in dirs
                  if (Path(d) / "saefree_by_arm.csv").exists()), None)
        if not f:
            record("4.3", f"probes at block 18 ({tag})", "not run", "-", "MISSING")
            continue
        rows = list(csv.DictReader(open(f)))
        cols = [c for c in rows[0] if c.startswith("probe_") and c.endswith("_f1")]
        causal_wins, total, detail = 0, 0, []
        for c in cols:
            d = []
            for s in SEEDS:
                try:
                    m = next(r for r in rows if r["arm"] == ARMS["mlm"].format(s))
                    k = next(r for r in rows if r["arm"] == ARMS["clm"].format(s))
                    d.append(float(m[c]) - float(k[c]))
                except (StopIteration, ValueError, KeyError):
                    pass
            if d:
                w = sum(1 for x in d if x < 0)
                causal_wins += w; total += len(d)
                detail.append(f"{c.replace('probe_','').replace('_f1','')} {w}/{len(d)}")
        record("4.3", f"probes at block 18 ({tag}), causal favoured",
               "27/27 at blocks 7/11/14", f"{causal_wins}/{total}",
               "CHANGED" if causal_wins < total else "PASS", " | ".join(detail))

    # ---- report ------------------------------------------------------------
    w = max(len(c) for _, c, _, _, _, _ in results) + 2
    cur = None
    for sec, claim, paper, got, verdict, note in results:
        if sec != cur:
            print(f"\n§{sec}" if sec[0].isdigit() else f"\n{sec}"); cur = sec
        print(f"  [{verdict:<7}] {claim:<{w}} paper: {paper:<26} computed: {got}")
        if note:
            print(f"            {note}")

    changed = [r for r in results if r[4] == "CHANGED"]
    missing = [r for r in results if r[4] == "MISSING"]
    print("\n" + "-" * 92)
    print(f"  {sum(1 for r in results if r[4]=='PASS')} pass, {len(changed)} changed, "
          f"{len(missing)} missing, {sum(1 for r in results if r[4]=='INFO')} new")
    if changed:
        print("\n  CHANGED — the paper says something this data does not:")
        for r in changed:
            print(f"    §{r[0]} {r[1]}: {r[2]} -> {r[3]}")
    if missing:
        print(f"\n  MISSING — not in this delivery: {len(missing)} claim(s)")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
