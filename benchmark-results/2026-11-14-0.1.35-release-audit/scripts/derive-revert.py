#!/usr/bin/env python3
"""Revert run: per arm, per-fork medians (avgt us/op, lower is better) over all rounds; ratio to v0.1.34 and v0.1.35."""
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1]); data = {}
for f in sorted(root.glob("*.json")):
    arm = f.name.rsplit("-searchRawHnsw-r", 1)[0]
    for rec in json.loads(f.read_text()):
        key = ",".join(f"{k}={v}" for k, v in sorted((rec.get("params") or {}).items()))
        data.setdefault(key, {}).setdefault(arm, []).extend(statistics.median(fk) for fk in rec["primaryMetric"]["rawData"])
order = ["v0.1.34", "v0.1.35", "fixA", "rev-871dafcc", "rev-73bcc1b8", "rev-97579da4", "rev-b500e86e"]
for key, d in data.items():
    med = {a: statistics.median(v) for a, v in d.items()}
    print(f"\nsearchRawHnsw {{{key}}} (avgt us/op, lower is better; per-fork medians)")
    print("| arm | forks | median | min | max | vs v0.1.34 | vs v0.1.35 | P(fork faster than a v0.1.35 fork) |"); print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    b35 = d.get("v0.1.35", [])
    for a in order + sorted(set(d) - set(order)):
        if a not in d: continue
        v = d[a]; wins = sum(1 for x in v for y in b35 if x < y) / (len(v) * len(b35)) if b35 else float("nan")
        print(f"| {a} | {len(v)} | {med[a]:.2f} | {min(v):.2f} | {max(v):.2f} | {med[a]/med['v0.1.34']:.3f} | {med[a]/med['v0.1.35']:.3f} | {wins:.2f} |")
