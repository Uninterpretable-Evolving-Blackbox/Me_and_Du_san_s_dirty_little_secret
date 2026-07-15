#!/usr/bin/env python3
"""
fetch_pdbs.py — download ONLY the PDB entries the 1,500 eval proteins need (~1.1 GB).

L_struct needs Ca coordinates, which come from these structures. We fetch just the
1,500 entries referenced by eval_set/uids.json (not all ~12k of SCOPe).

Idempotent: existing non-empty files are skipped, so re-running resumes.
No DSSP required — the DSSP-derived features already ship in cache/residue_features.csv.
"""
import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://files.rcsb.org/download/{}.pdb"
UA = {"User-Agent": "controlled-plm-study/1.0 (academic use; contact via repo)"}


def pdb_ids(eval_set):
    uids = json.loads((Path(eval_set) / "uids.json").read_text())
    return sorted({str(u)[1:5].lower() for u in uids})


def fetch(pid, out_dir, retries=3):
    f = Path(out_dir) / f"{pid}.pdb"
    if f.exists() and f.stat().st_size > 0:
        return "skip"
    for _ in range(retries):
        try:
            req = urllib.request.Request(URL.format(pid), headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if data:
                tmp = f.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.rename(f)          # atomic: never leave a truncated .pdb
                return "ok"
        except Exception:
            continue
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set")
    ap.add_argument("--out-dir", default="cache/pdb_files")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ids = pdb_ids(args.eval_set)
    print(f"need {len(ids)} PDB entries -> {args.out_dir}")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, p, args.out_dir): p for p in ids}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            counts[r] += 1
            if r == "fail":
                failed.append(futs[fut])
            if i % 200 == 0:
                print(f"  {i}/{len(ids)}  ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")

    print(f"\ndone: ok={counts['ok']} skipped={counts['skip']} failed={counts['fail']}")
    if failed:
        print("FAILED ids:", " ".join(failed[:20]), "..." if len(failed) > 20 else "")
        print("Re-run this script to retry (it resumes).")
        sys.exit(1)
    print("All eval structures present.")


if __name__ == "__main__":
    main()
