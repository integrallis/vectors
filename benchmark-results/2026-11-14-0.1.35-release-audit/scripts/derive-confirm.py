#!/usr/bin/env python3
"""Confirm run: per arm and benchmark, median over both rounds' per-fork medians; ratio to v0.1.34 and v0.1.35."""
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1]); arms = ["v0.1.34", "v0.1.35", "fixA", "fixB"]
data = {}
for f in sorted(root.glob("*.json")):
    arm, rest = f.name.split("-", 1); stem = rest.rsplit("-r", 1)[0]
    for rec in json.loads(f.read_text()):
        params = rec.get("params") or {}
        key = stem + (("{" + ",".join(f"{k}={v}" for k, v in sorted(params.items())) + "}") if params else "")
        d = data.setdefault(key, {"mode": rec["mode"], "unit": rec["primaryMetric"]["scoreUnit"], "arms": {}})
        d["arms"].setdefault(arm, []).extend(statistics.median(fk) for fk in rec["primaryMetric"]["rawData"])
for key, d in data.items():
    hi = d["mode"] == "thrpt"
    print(f"\n{key}  ({d['mode']}, {d['unit']}; per-fork medians, n per arm)")
    print("| arm | median | forks | vs v0.1.34 | vs v0.1.35 |"); print("| --- | --- | --- | --- | --- |")
    med = {a: statistics.median(v) for a, v in d["arms"].items() if v}
    for a in arms:
        if a not in med: continue
        r34 = med[a] / med["v0.1.34"]; r35 = med[a] / med["v0.1.35"]
        def word(r): 
            ch = (r - 1) if hi else (1 - r); return f"{r:.3f} ({'+' if ch >= 0 else ''}{ch*100:.1f}%)"
        print(f"| {a} | {med[a]:.3f} | {', '.join(f'{x:.2f}' for x in d['arms'][a])} | {word(r34)} | {word(r35)} |")
