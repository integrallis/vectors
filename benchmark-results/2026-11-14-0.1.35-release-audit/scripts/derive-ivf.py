#!/usr/bin/env python3
"""IVF recall-vs-latency: (a) TieredIvfRecallQpsBenchmark logs  <tag>-dim<d>-n<n>-r<r>.log with rows
'nprobe recall@10 qps latency_us'; (b) RecallQpsBenchmark recall-qps.csv per tag (fashion-mnist, ivf_flat).
Prints, per shape and nprobe, recall and latency for both tags side by side (latency = median across rounds)."""
import csv, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
cur = root / "ivfcurve"
if cur.is_dir():
    data = {}
    for f in sorted(cur.glob("*.log")):
        m = re.match(r"(v[\d.]+)-dim(\d+)-n(\d+)-r(\d+)\.log", f.name)
        if not m: continue
        tag, dim, n, r = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        build = next((l for l in f.read_text().splitlines() if "built tiered IVF" in l), "")
        for l in f.read_text().splitlines():
            mm = re.match(r"^(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$", l)
            if mm:
                d = data.setdefault((dim, n), {}).setdefault(int(mm.group(1)), {}).setdefault(tag, {"recall": [], "qps": [], "lat": [], "build": set()})
                d["recall"].append(float(mm.group(2))); d["qps"].append(float(mm.group(3))); d["lat"].append(float(mm.group(4))); d["build"].add(build.strip())
    for (dim, n), rows in sorted(data.items()):
        print(f"\n### Tiered IVF (BuoyIndex) synthetic mixture-of-Gaussians, dim={dim} n={n:,}, recall@10 vs median latency (us), both rounds")
        print("| nprobe | recall 0.1.34 | recall 0.1.35 | Δrecall | latency 0.1.34 | latency 0.1.35 | 0.1.35/0.1.34 |"); print("| --- | --- | --- | --- | --- | --- | --- |")
        for np_, arms in sorted(rows.items()):
            a, b = arms.get("v0.1.34"), arms.get("v0.1.35")
            if not a or not b: continue
            ra, rb = statistics.median(a["recall"]), statistics.median(b["recall"]); la, lb = statistics.median(a["lat"]), statistics.median(b["lat"])
            print(f"| {np_} | {ra:.4f} | {rb:.4f} | {rb-ra:+.4f} | {la:.1f} | {lb:.1f} | {lb/la:.3f} |")
        for tag in ("v0.1.34", "v0.1.35"):
            b = {x for arms in rows.values() for t, d in arms.items() if t == tag for x in d["build"]}
            for x in sorted(b): print(f"    {tag}: {x}")
real = root / "realivf"
if real.is_dir():
    tbl = {}
    for tag in ("v0.1.34", "v0.1.35"):
        p = real / tag / "recall-qps.csv"
        if not p.is_file(): print(f"\n(realivf: no csv for {tag})"); continue
        with p.open() as fh:
            rd = csv.DictReader(fh); cols = rd.fieldnames
            for row in rd:
                if "ivf" not in (row.get("algorithm") or row.get("algo") or "").lower(): continue
                key = (row.get("dataset"), row.get("build_params") or row.get("buildParams"), row.get("search_params") or row.get("searchParams"))
                tbl.setdefault(key, {})[tag] = row
    if tbl:
        sample = next(iter(tbl.values())); anyrow = next(iter(sample.values()))
        rc = next((c for c in anyrow if "recall" in c.lower()), None); qc = next((c for c in anyrow if c.lower().startswith("qps")), None); lc = next((c for c in anyrow if "p50" in c.lower() or "median" in c.lower()), None)
        print(f"\n### Real data: {next(iter(tbl))[0]} IVF_FLAT (RecallQpsBenchmark), columns {rc}/{qc}/{lc}")
        print("| build | search | recall 0.1.34 | recall 0.1.35 | Δ | qps 0.1.34 | qps 0.1.35 | 0.1.35/0.1.34 |"); print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key in sorted(tbl, key=lambda k: (str(k[1]), int(re.sub(r"\D", "", str(k[2])) or 0))):
            a, b = tbl[key].get("v0.1.34"), tbl[key].get("v0.1.35")
            if not a or not b: continue
            ra, rb = float(a[rc]), float(b[rc]); qa, qb = float(a[qc]), float(b[qc])
            print(f"| {key[1]} | {key[2]} | {ra:.4f} | {rb:.4f} | {rb-ra:+.4f} | {qa:.0f} | {qb:.0f} | {qb/qa:.3f} |")
    else:
        print("\n(realivf: no ivf rows parsed; csv columns:", cols if 'cols' in dir() else None, ")")
