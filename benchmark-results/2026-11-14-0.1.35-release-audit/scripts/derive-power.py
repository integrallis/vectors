#!/usr/bin/env python3
"""Power run: per arm and shape, the per-fork medians of all rounds (n=15), their median, min, max,
and the ratio to v0.1.34 / v0.1.35. Also the fraction of pairwise fork comparisons where the arm beats v0.1.34."""
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1]); arms = ["v0.1.34", "v0.1.35", "fixA"]
data = {}
for f in sorted(root.glob("*.json")):
    arm = f.name.split("-")[0]
    for rec in json.loads(f.read_text()):
        n = rec["params"]["n"]
        data.setdefault(n, {}).setdefault(arm, []).extend(statistics.median(fk) for fk in rec["primaryMetric"]["rawData"])
for n in sorted(data, key=int):
    d = data[n]; med = {a: statistics.median(v) for a, v in d.items()}
    print(f"\nsearchHnsw n={n} dim=128 k=10 (thrpt ops/ms, per-fork medians)")
    print("| arm | forks | median | min | max | vs v0.1.34 | vs v0.1.35 | P(fork beats a v0.1.34 fork) |"); print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    base = d.get("v0.1.34", [])
    for a in arms:
        if a not in d: continue
        v = d[a]; wins = sum(1 for x in v for y in base if x > y) / (len(v) * len(base)) if base else float("nan")
        print(f"| {a} | {len(v)} | {med[a]:.2f} | {min(v):.2f} | {max(v):.2f} | {med[a]/med['v0.1.34']:.3f} | {med[a]/med['v0.1.35']:.3f} | {wins:.2f} |")
