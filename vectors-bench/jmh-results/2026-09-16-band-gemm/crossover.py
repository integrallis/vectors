#!/usr/bin/env python3
"""Crossover analysis for pre-registration 2 (batch-dispatched hybrid).

Usage: crossover.py raw/<label>/crossover.json [...]
Derived from the coordinator's cross.py (same arithmetic), parameterised by path. Speedup =
integer ms / band ms; '*' = band ahead beyond both error bars (band + err < integer - err).
T per cell = smallest batch from which every larger measured batch is '*'.
"""
import collections, json, sys

for path in sys.argv[1:]:
    runs = json.load(open(path))
    cells = collections.defaultdict(dict)
    for b in runs:
        p, m = b["params"], b["primaryMetric"]
        cells[(p["format"], p["shape"], int(p["batchSize"]))][p["kernel"]] = (m["score"], m["scoreError"])
    batches = sorted({k[2] for k in cells})
    print(f"\n### {path}  (forks={runs[0]['forks']}, jvmArgs={' '.join(runs[0]['jvmArgs'])}, jdk={runs[0]['jdkVersion']})\n")
    print("| format | shape | " + " | ".join(str(b) for b in batches) + " | T |")
    print("|---|---|" + "---:|" * (len(batches) + 1))
    for f in sorted({k[0] for k in cells}):
        for sh in sorted({k[1] for k in cells}):
            row, t = [], None
            for bt in batches:
                c = cells.get((f, sh, bt), {})
                if "BAND_F32" not in c or "INTEGER" not in c:
                    row.append("-")
                    continue
                (bs, be), (i, ie) = c["BAND_F32"], c["INTEGER"]
                clear = bs + be < i - ie
                row.append(f"{i / bs:.2f}{'*' if clear else ''}")
                if clear and t is None:
                    t = bt
                if t is not None and not clear:
                    t = None
            print(f"| {f} | {sh} | " + " | ".join(row) + f" | {t} |")
