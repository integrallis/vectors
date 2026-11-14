#!/usr/bin/env python3
"""Phase 3: KMeansEMeansBenchmark on v0.1.35 — trainKMeans (50 Lloyd iterations, the 0.1.34 BuoyIndex call) vs
trainEMeans (5 epochs, the 0.1.35 call) on the same data; avgt ms/op, median over iterations; ratio e-means/k-means."""
import json, pathlib, statistics, sys
f = pathlib.Path(sys.argv[1]) / "v0.1.35-KMeansEMeansBenchmark.json"
rows = {}
for rec in json.loads(f.read_text()):
    p = rec["params"]; key = (int(p["n"]), int(p["dim"]), int(p["k"])); m = rec["benchmark"].split(".")[-1]
    rows.setdefault(key, {})[m] = statistics.median([x for fk in rec["primaryMetric"]["rawData"] for x in fk])
print("| n | dim | k | trainKMeans ms | trainEMeans ms | e-means / k-means |"); print("| --- | --- | --- | --- | --- | --- |")
ratios = []
for key in sorted(rows):
    a, b = rows[key]["trainKMeans"], rows[key]["trainEMeans"]; ratios.append(b / a)
    print(f"| {key[0]:,} | {key[1]} | {key[2]} | {a:.1f} | {b:.1f} | {b/a:.3f} |")
print(f"\nratio range {min(ratios):.3f}..{max(ratios):.3f}, median {statistics.median(ratios):.3f} over {len(ratios)} shapes")
