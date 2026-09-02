#!/usr/bin/env python3
"""
Smoke test for verify_paper_claims.py.

A verifier that cannot fail is worse than none: it converts "unchecked" into
"checked and fine". So these fixtures are built to break it — a perturbed value,
a missing tree, two archives disagreeing about the same cell — and the test is
that each one is reported rather than absorbed.

    python3 tests/test_verify_paper_claims.py
"""
import csv
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verify_paper_claims.py"
FAILURES = []

# the paper's Table 2 trained means, and the order-destroyed level, per (arm, layer)
NATIVE = {("mlm", 11): 0.01881, ("mlm", 14): 0.01667, ("mlm", 18): 0.01417,
          ("clm", 11): 0.00162, ("clm", 14): 0.00144, ("clm", 18): 0.00413}
SHUF = {("mlm", 11): 0.02730, ("mlm", 14): 0.02730, ("mlm", 18): 0.02730,
        ("clm", 11): 0.05201, ("clm", 14): 0.05201, ("clm", 18): 0.05201}
ARM = {"mlm": "ckpt_mlm_s{}_token", "clm": "ckpt_clm_s{}"}


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def write_cell(path, value, cols=("struct_delta",)):
    path.mkdir(parents=True, exist_ok=True)
    f = path / "struct_seq_metrics.csv"
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_idx", *cols])
        # two features whose mean is exactly `value`
        w.writerow([0, f"{value - 0.001:.8f}"])
        w.writerow([1, f"{value + 0.001:.8f}"])
    return f


def build(base, native=NATIVE, shuf=SHUF, denoms=True):
    """A minimal delivery that should verify clean."""
    for tree, table in (("outputs_ctrl", native), ("outputs_ctrl_shuf", shuf)):
        for (ak, L), v in table.items():
            for s in ("42", "43", "44"):
                d = Path(base) / "arch" / tree / ARM[ak].format(s) / f"layer_{L}"
                write_cell(d, v)
                if denoms:
                    with open(d / "struct_seq_metrics_denominators.csv", "w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["feature_idx"] + [f"struct_delta_{m}"
                                                      for m in ("sd", "fixed", "iqr", "rank")])
                        w.writerow([0] + [f"{v:.8f}"] * 4)
    return Path(base) / "arch"


def run(d):
    r = subprocess.run([sys.executable, str(SCRIPT), "--results", str(d)],
                       capture_output=True, text=True, timeout=180)
    return r.returncode, r.stdout + r.stderr


def main():
    print("=" * 68)
    print("test_verify_paper_claims.py")
    print("=" * 68)

    # ---- clean delivery verifies -----------------------------------------
    with tempfile.TemporaryDirectory() as t:
        d = build(t)
        rc, out = run(d)
        check("a clean delivery exits 0", rc == 0, f"rc={rc}")
        check("  and reproduces Table 2", out.count("Table 2") and "CHANGED" not in out.split("Table 2")[1][:600])
        check("  and reports 18/18", "computed: 18/18" in out)
        check("  and reports Delta", "+0.01415" in out)

    # ---- a perturbed value must be caught --------------------------------
    with tempfile.TemporaryDirectory() as t:
        bad = dict(NATIVE); bad[("mlm", 14)] = 0.01900        # paper says 0.01667
        d = build(t, native=bad)
        rc, out = run(d)
        check("a wrong Table 2 value exits 1", rc == 1, f"rc={rc}")
        check("  and is named in the CHANGED list", "trained MLM block 14" in out)
        check("  showing both numbers", "+0.01667" in out and "+0.01900" in out)

    # ---- a sign flip in the corpus control must be caught ----------------
    with tempfile.TemporaryDirectory() as t:
        # derive from NATIVE so every shuffled cell is unambiguously below its
        # own native counterpart; scaling SHUF alone leaves the causal cells above
        low = {k: v / 10 for k, v in NATIVE.items()}
        d = build(t, shuf=low)
        rc, out = run(d)
        check("a reversed corpus control exits 1", rc == 1, f"rc={rc}")
        check("  and reports 0/18 rather than 18/18", "computed: 0/18" in out)

    # ---- missing trees report MISSING, never PASS ------------------------
    with tempfile.TemporaryDirectory() as t:
        d = build(t)
        shutil.rmtree(d / "outputs_ctrl_shuf")
        rc, out = run(d)
        check("a missing tree is reported MISSING", "[MISSING" in out)
        check("  and is not silently passed", "computed: 18/18" not in out)

    # ---- two archives disagreeing about one cell -------------------------
    with tempfile.TemporaryDirectory() as t:
        d = build(t)
        second = Path(t) / "arch2" / "outputs_ctrl" / ARM["mlm"].format("42") / "layer_14"
        write_cell(second, 0.09999)                            # same cell, different value
        rc, out = run(Path(t))
        check("cells that disagree across archives are caught",
              "clash" in out and "[CHANGED" in out.split("consistency")[1][:400])

    # ---- .tgz archives are unpacked --------------------------------------
    with tempfile.TemporaryDirectory() as t:
        d = build(t)
        tgz_dir = Path(t) / "packed"; tgz_dir.mkdir()
        with tarfile.open(tgz_dir / "batch.tgz", "w:gz") as tf:
            tf.add(d, arcname="arch")
        rc, out = run(tgz_dir)
        check("a directory of .tgz files is unpacked", "unpacked 1 archive" in out, f"rc={rc}")
        check("  and verifies from them", "computed: 18/18" in out)

    # ---- an empty or absent directory ------------------------------------
    with tempfile.TemporaryDirectory() as t:
        rc, out = run(Path(t))
        check("an empty directory does not report success", rc != 0 or "MISSING" in out)
    rc, out = run(Path("/nonexistent/xyz"))
    check("a missing directory exits 2", rc == 2, f"rc={rc}")

    # ---- the denominator check needs all four modes ----------------------
    with tempfile.TemporaryDirectory() as t:
        d = build(t, denoms=False)
        rc, out = run(d)
        check("absent denominator files report MISSING",
              "denominator variants" in out and "[MISSING" in out)

    print("-" * 68)
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
