#!/usr/bin/env python3
"""check_grid.py — audit the InterPLM grid before anyone reads an F1 number.

Reads only. No compute, no GPU, seconds to run. Two questions:

  1. COMPLETENESS. Which (arm, model-seed, layer, SAE-seed) cells produced a dictionary and
     which produced a concept-F1 block? The grid runs masked seeds before causal ones inside
     each layer instance, so a partial grid is always masked-heavy — which makes an
     arm comparison look available long before it is.

  2. COMPARABILITY. The project's outcome-blind rule (PIPELINE_REFERENCE.md §7): if the two
     arms' live-feature counts or L0 differ substantially, the between-arm F1 gap is not
     interpretable and the non-comparability IS the result. This prints the gate rather than
     leaving it to be noticed.

Usage:
    python check_grid.py                       # $BASE/results, or --base
    python check_grid.py --base ~/interplm_stress
    python check_grid.py --strict              # exit 1 when the grid is not usable

Inputs, both written by RUN_INTERPLM_STRESS.sh:
    $BASE/results/sae_quality.txt   tag var_expl dead live L0 max_frac
    $BASE/results/concept_f1.txt    ----- tag ----- blocks
"""
import argparse
import os
import re
import sys
from collections import defaultdict

# Tags the grid actually emits:
#   clm_s42_L11_s0        native causal          shuf_clm_s42_L11_s0   shuffled causal
#   mlm_s42_L11_s0        native masked          shuf_mlm_s42_L11_s0   shuffled masked
#   untrained_L11_s0      untrained baseline (no model seed)
# A tag that does not match is REPORTED, never dropped silently — a partial parse that
# quietly ignores a third of the grid is worse than no parse at all.
TAG_RE = re.compile(
    r"^(?P<arm>(?:shuf_)?(?:clm|mlm)|untrained)"
    r"(?:_s(?P<seed>\d+))?_L(?P<layer>\d+)_s(?P<sae>\d+)$"
)

# The two arms whose comparison the gate is about. Shuffled and untrained cells are
# controls, not arms, and must not be pooled into either side.
NATIVE_ARMS = ("mlm", "clm")


def parse_quality(path):
    """(tag -> quality dict, list of unrecognised tags)."""
    out, unknown = {}, []
    if not os.path.exists(path):
        return out, unknown
    for line in open(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        if not TAG_RE.match(parts[0]):
            unknown.append(parts[0])
            continue
        try:
            out[parts[0]] = {
                "var_ev": float(parts[1]), "dead": int(parts[2]),
                "live": int(parts[3]), "l0": float(parts[4]),
            }
        except ValueError:
            unknown.append(parts[0])
    return out, unknown


def parse_f1(path):
    """tag -> {avg_f1, n_concepts, n_features}."""
    out, tag = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"^-+\s*(\S+)\s*-+$", line.strip())
        if m:
            tag = m.group(1)
            out.setdefault(tag, {})
            continue
        if tag is None:
            continue
        m = re.search(r"Average best F1 per concept in test set:\s*([\d.]+)", line)
        if m:
            out[tag]["avg_f1"] = float(m.group(1))
        m = re.search(r"Number of concepts identified:\s*(\d+)", line)
        if m:
            out[tag]["n_concepts"] = int(m.group(1))
    return out


def split(tag):
    m = TAG_RE.match(tag)
    return m.groupdict() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", os.path.expanduser("~/interplm_stress")))
    ap.add_argument("--results", default=None, help="overrides <base>/results")
    ap.add_argument("--expect-cells", type=int, default=81,
                    help="grid size, for the completeness line only (9 models x 3 layers x 3 SAE seeds)")
    ap.add_argument("--l0-ratio", type=float, default=1.5,
                    help="flag when one arm's median L0 exceeds the other's by this factor")
    ap.add_argument("--live-ratio", type=float, default=1.15,
                    help="flag when one arm's median live-feature count exceeds the other's by this factor")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the grid is not usable")
    a = ap.parse_args()

    res = a.results or os.path.join(a.base, "results")
    qual, unknown = parse_quality(os.path.join(res, "sae_quality.txt"))
    f1 = parse_f1(os.path.join(res, "concept_f1.txt"))

    print(f"results dir: {res}")
    if not qual:
        print(f"  NO sae_quality.txt rows found under {res} — nothing to audit.")
        print("  That is a FAILURE, not an empty pass: either --base is wrong or the grid")
        print("  wrote nothing. Check the path before reading anything else as a result.")
        return 1

    if unknown:
        print()
        print(f"  UNRECOGNISED TAGS: {len(unknown)} row(s) could not be parsed and are NOT")
        print(f"  counted below — fix TAG_RE before trusting any number here.")
        for t in sorted(set(unknown))[:8]:
            print(f"    {t}")

    # ---------------------------------------------------------------- completeness
    by_arm = defaultdict(list)
    for tag in qual:
        d = split(tag)
        if d:
            by_arm[d["arm"]].append(tag)
    print()
    print(f"  SAEs trained : {len(qual)} / {a.expect_cells}")
    print(f"  F1 scored    : {len(f1)} / {a.expect_cells}")
    print()
    print("  %-10s %6s %6s   %s" % ("arm", "SAEs", "F1", "cells present (seed/layer)"))
    for arm in sorted(by_arm):
        tags = sorted(by_arm[arm])
        cells = sorted({(split(t)["seed"] or "-", split(t)["layer"]) for t in tags})
        shown = " ".join(f"s{s}L{l}" for s, l in cells[:8])
        more = "" if len(cells) <= 8 else f" (+{len(cells)-8} more)"
        n_f1 = sum(1 for t in tags if t in f1)
        print("  %-10s %6d %6d   %s%s" % (arm, len(tags), n_f1, shown, more))

    arms = [x for x in NATIVE_ARMS if x in by_arm]
    controls = sorted(set(by_arm) - set(NATIVE_ARMS))
    if controls:
        print(f"\n  controls present (not pooled into either arm): {', '.join(controls)}")
    if len(arms) < 2:
        print("\n  FEWER THAN TWO NATIVE ARMS — no between-arm comparison is possible yet.")
        return 1 if a.strict else 0

    # ---------------------------------------------------------------- imbalance
    counts = {arm: len(by_arm[arm]) for arm in arms}
    lo, hi = min(counts, key=counts.get), max(counts, key=counts.get)
    print()
    if counts[lo] == 0:
        print(f"  ARM MISSING: {lo} has no dictionaries. The grid is not comparable.")
        return 1 if a.strict else 0
    if counts[hi] > 2 * counts[lo]:
        print(f"  ARM IMBALANCE: {hi}={counts[hi]} vs {lo}={counts[lo]} dictionaries.")
        print(f"  Each layer instance trains masked seeds before causal ones, so a partial grid")
        print(f"  is masked-heavy by construction. Do not read the arm gap from this state.")

    # ---------------------------------------------------------------- comparability gate
    def med(vals):
        v = sorted(vals)
        return v[len(v) // 2] if v else float("nan")

    print()
    print("  COMPARABILITY GATE (medians over the cells that exist)")
    print("  %-6s %8s %8s %8s %8s" % ("arm", "var_EV", "dead", "live", "L0"))
    stats = {}
    for arm in arms:
        rows = [qual[t] for t in by_arm[arm]]
        stats[arm] = {k: med([r[k] for r in rows]) for k in ("var_ev", "dead", "live", "l0")}
        s = stats[arm]
        print("  %-6s %8.4f %8.0f %8.0f %8.1f" % (arm, s["var_ev"], s["dead"], s["live"], s["l0"]))

    a0, a1 = arms[0], arms[1]
    def ratio(k):
        x, y = stats[a0][k], stats[a1][k]
        lo_, hi_ = min(x, y), max(x, y)
        return hi_ / lo_ if lo_ > 0 else float("inf")

    problems = []
    if ratio("live") > a.live_ratio:
        problems.append(f"live features differ by {ratio('live'):.2f}x (gate {a.live_ratio}x)")
    if ratio("l0") > a.l0_ratio:
        problems.append(f"L0 differs by {ratio('l0'):.2f}x (gate {a.l0_ratio}x)")

    print()
    if problems:
        print("  GATE FAILED:")
        for p in problems:
            print(f"    - {p}")
        print("  The arms do not hold comparable dictionaries, so the between-arm F1 gap is")
        print("  NOT interpretable as a property of the models. Report the incomparability as")
        print("  the finding and send sae_quality.txt alongside — never concept_f1.txt alone.")
    else:
        print("  GATE PASSED: the two arms hold comparable dictionaries on live count and L0.")

    if f1:
        print()
        print("  mean best-F1 by arm (only for cells that are scored):")
        for arm in arms:
            vals = [f1[t]["avg_f1"] for t in by_arm[arm] if t in f1 and "avg_f1" in f1[t]]
            if vals:
                print("    %-6s n=%-3d mean=%.4f  [%.4f-%.4f]"
                      % (arm, len(vals), sum(vals) / len(vals), min(vals), max(vals)))
            else:
                print(f"    {arm:<6} no scored cells")

    usable = not problems and counts[hi] <= 2 * counts[lo] and len(qual) >= a.expect_cells
    print()
    print("  VERDICT: " + ("grid is complete and comparable" if usable
                           else "grid is NOT yet usable for an arm comparison"))
    return 0 if usable or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
