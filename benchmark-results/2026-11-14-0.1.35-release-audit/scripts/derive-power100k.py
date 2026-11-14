#!/usr/bin/env python3
"""power100k: per benchmark method, per-fork medians (avgt us/op) of all rounds for both tags; median, min, max, ratio, P(0.1.35 fork faster than a 0.1.34 fork)."""
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1]); data = {}
for f in sorted(root.glob("*.json")):
    tag = f.name.split("-")[0]
    for rec in json.loads(f.read_text()):
        key = rec["benchmark"].split(".")[-1] + "{" + ",".join(f"{k}={v}" for k, v in sorted((rec.get("params") or {}).items())) + "}"
        data.setdefault(key, {}).setdefault(tag, []).extend(statistics.median(fk) for fk in rec["primaryMetric"]["rawData"])
print("| benchmark (100k x 128) | v0.1.34 median [min..max] (forks) | v0.1.35 median [min..max] (forks) | 0.1.35 / 0.1.34 | P(0.1.35 fork faster) |"); print("| --- | --- | --- | --- | --- |")
for key, d in sorted(data.items()):
    a, b = d.get("v0.1.34", []), d.get("v0.1.35", [])
    if not a or not b: continue
    p = sum(1 for x in b for y in a if x < y) / (len(a) * len(b))
    print(f"| {key} | {statistics.median(a):.2f} [{min(a):.2f}..{max(a):.2f}] ({len(a)}) | {statistics.median(b):.2f} [{min(b):.2f}..{max(b):.2f}] ({len(b)}) | {statistics.median(b)/statistics.median(a):.3f} | {p:.2f} |")
