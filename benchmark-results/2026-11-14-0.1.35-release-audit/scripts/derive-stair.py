#!/usr/bin/env python3
"""Staircase: per point, median of the per-fork medians of AnnSearchBenchmark.searchHnsw (ops/ms),
both rounds, plus the ratio to the v0.1.34 point and the step from the previous point."""
import json, pathlib, statistics, sys, subprocess
root = pathlib.Path(sys.argv[1]); repo = sys.argv[2] if len(sys.argv) > 2 else None
points = (root / "points.txt").read_text().split()
rows = []
for i, p in enumerate(points):
    forks = {}
    for f in sorted(root.glob(f"{i:02d}-{p}-r*.json")):
        rec = json.loads(f.read_text())[0]
        forks[f.stem.split("-r")[-1]] = [statistics.median(fk) for fk in rec["primaryMetric"]["rawData"]]
    allf = [x for v in forks.values() for x in v]
    rows.append((i, p, forks, statistics.median(allf) if allf else None))
base = rows[0][3]
subj = {}
if repo:
    for _, p, _, _ in rows:
        try: subj[p] = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%s", p], capture_output=True, text=True).stdout.strip()[:70]
        except Exception: pass
print("| # | point | r1 forks | r2 forks | median | vs v0.1.34 | step | subject |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
prev = None
for i, p, forks, med in rows:
    r1 = ", ".join(f"{x:.2f}" for x in forks.get("1", [])); r2 = ", ".join(f"{x:.2f}" for x in forks.get("2", []))
    if med is None: print(f"| {i} | {p} | {r1} | {r2} | – | – | – | {subj.get(p,'')} |"); continue
    step = (med / prev - 1) * 100 if prev else 0.0
    print(f"| {i} | {p} | {r1} | {r2} | {med:.2f} | {med/base:.3f} | {step:+.1f}% | {subj.get(p,'')} |")
    prev = med
