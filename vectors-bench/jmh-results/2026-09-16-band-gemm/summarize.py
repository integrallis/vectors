#!/usr/bin/env python3
"""Summarize band GEMM A/B JMH JSON into Markdown tables.

Usage: summarize.py raw/<label>
Adds, per row: integer and band ms/op, band speedup (integer ms / band ms), quantized weight bytes
read per second (the matrix is read once per op by both arms), that rate as a % of the host's
measured sequential-read bandwidth (single-thread probe for -Dvectors.gguf.parallel=false runs,
all-threads probe otherwise), and multiply-adds per second (batch*rows*cols / s).
"""
import json, os, sys, glob

BLOCK = {"Q4_K": (256, 144), "Q6_K": (256, 210), "Q8_0": (32, 34)}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []  # missing or still being written


def bandwidth(d, name):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p):
        return None
    out = {}
    for r in load(p):
        size_mb = int(r["params"]["sizeMb"])
        ms = r["primaryMetric"]["score"]
        threads = r["threads"]
        gbps = size_mb * 2**20 / 1e9 / (ms / 1000.0) * threads
        out[r["benchmark"].split(".")[-1]] = (gbps, threads)
    return out


def ab_rows(d, name):
    p = os.path.join(d, name + ".json")
    if not os.path.exists(p):
        return {}
    rows = {}
    for r in load(p):
        pr = r["params"]
        key = (pr["format"], pr["shape"], int(pr["batchSize"]), pr.get("storage", "heap"))
        rows.setdefault(key, {})[pr["kernel"]] = (r["primaryMetric"]["score"], r["primaryMetric"]["scoreError"])
    return rows


def table(title, rows, bw):
    print(f"\n### {title}\n")
    print("| format | shape (rows x cols) | batch | storage | integer ms/op | band ms/op | band speedup | weights GB/s int / band | % bandwidth int / band | mul-adds/s int / band (G) |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    speedups = []
    for key in sorted(rows, key=lambda k: (k[1], k[0], k[2], k[3])):
        fmt, shape, batch, storage = key
        arms = rows[key]
        if "INTEGER" not in arms or "BAND_F32" not in arms:
            continue
        m, k = map(int, shape.split("x"))
        bs, bb = BLOCK[fmt]
        wbytes = m * (k // bs) * bb
        (ims, ie), (bms, be) = arms["INTEGER"], arms["BAND_F32"]
        sp = ims / bms
        speedups.append((key, sp))
        igb, bgb = wbytes / 1e9 / (ims / 1e3), wbytes / 1e9 / (bms / 1e3)
        pct = f"{100*igb/bw:.1f} / {100*bgb/bw:.1f}" if bw else "n/a"
        ma = batch * m * k
        print(f"| {fmt} | {shape} | {batch} | {storage} | {ims:.3f} ± {ie:.3f} | {bms:.3f} ± {be:.3f} | **{sp:.2f}x** | {igb:.2f} / {bgb:.2f} | {pct} | {ma/(ims/1e3)/1e9:.2f} / {ma/(bms/1e3)/1e9:.2f} |")
    return speedups


def main():
    d = sys.argv[1]
    bw1 = bandwidth(d, "bandwidth-t1")
    bwall = bandwidth(d, "bandwidth-tall")
    print(f"## {os.path.basename(d.rstrip('/'))}")
    if bw1:
        print("\n### Sequential read bandwidth (MemoryBandwidthBenchmark)\n")
        print("| probe | threads | GB/s |\n|---|---:|---:|")
        for name, res in (("1 thread", bw1), ("all threads", bwall or {})):
            for bench, (g, t) in sorted(res.items()):
                print(f"| {bench} ({name}) | {t} | {g:.2f} |")
    st_bw = bw1["sequentialLongSum"][0] if bw1 else None
    mt_bw = bwall["sequentialLongSum"][0] if bwall else None
    all_sp = []
    for name, title, bw in (
        ("ab-mt-existing", "Multi-threaded, existing benchmark shape", mt_bw),
        ("ab-mt-llm", "Multi-threaded, Granite 4.1 3B FFN shapes", mt_bw),
        ("ab-st-existing", "Single-threaded (-Dvectors.gguf.parallel=false), existing shape", st_bw),
        ("ab-st-llm", "Single-threaded, Granite 4.1 3B FFN shapes", st_bw),
        ("ab-mt-mapped", "Multi-threaded, mapped weights (Q4_K)", mt_bw),
    ):
        rows = ab_rows(d, name)
        if rows:
            for key, sp in table(title, rows, bw):
                all_sp.append((name, key, sp))
    tiles = {}
    for tile in ("3x3", "4x4"):
        for key, arms in ab_rows(d, "tile-" + tile).items():
            tiles.setdefault(key, {})[tile] = arms["BAND_F32"][0]
    if tiles:
        print("\n### Band tile shape (multi-threaded, band arm only)\n")
        print("| format | shape | batch | 3x3 ms/op | 4x4 ms/op | 4x4 speedup |\n|---|---|---:|---:|---:|---:|")
        for key in sorted(tiles):
            t = tiles[key]
            if "3x3" in t and "4x4" in t:
                print(f"| {key[0]} | {key[1]} | {key[2]} | {t['3x3']:.3f} | {t['4x4']:.3f} | {t['3x3']/t['4x4']:.2f}x |")
    dq = os.path.join(d, "dequant.json")
    if os.path.exists(dq):
        print("\n### Band dequantization only (single thread)\n")
        print("| format | shape | ms/op | F32 elements/s (M) | quantized GB/s | % of 1-thread bandwidth |\n|---|---|---:|---:|---:|---:|")
        for r in load(dq):
            fmt, shape = r["params"]["format"], r["params"]["shape"]
            m, k = map(int, shape.split("x"))
            bs, bb = BLOCK[fmt]
            ms = r["primaryMetric"]["score"]
            gb = m * (k // bs) * bb / 1e9 / (ms / 1e3)
            pct = f"{100*gb/st_bw:.1f}" if st_bw else "n/a"
            print(f"| {fmt} | {shape} | {ms:.2f} | {m*k/(ms/1e3)/1e6:.0f} | {gb:.3f} | {pct} |")
    llm = [x for x in all_sp if x[1][1] in ("8192x2560", "2560x8192") and x[1][3] == "heap"]
    wins = [x for x in llm if x[2] >= 1.10]
    losses = [x for x in all_sp if x[2] < 1 / 1.05]
    print("\n### Pre-registered decision rule on this host\n")
    print(f"- LLM-shape configurations with band >= 1.10x: {len(wins)} of {len(llm)}")
    print(f"- configurations anywhere with band slower by > 5% (speedup < 0.952x): {len(losses)} of {len(all_sp)}")
    for name, key, sp in losses:
        print(f"  - {name} {key[0]} {key[1]} batch={key[2]} {key[3]}: {sp:.2f}x")
    verdict = "PASSES" if wins and not losses else "FAILS"
    print(f"- verdict for this ISA: **{verdict}** (>=10% LLM-shape win AND no >5% loss elsewhere)")


if __name__ == "__main__":
    main()
