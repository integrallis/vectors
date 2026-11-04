#!/usr/bin/env python3
"""Derive the thread-sweep table from raw logs. Nothing here is hand-typed."""
import re, sys, statistics as st
from pathlib import Path

def arms(d):
    out = {}
    for log in sorted(Path(d).glob("t*-r*.log")):
        m = re.match(r"t(\d+)-r(\w+)\.log", log.name)
        if not m: continue
        threads = int(m.group(1))
        text = log.read_text(errors="replace")
        p50 = re.search(r"warm: p50 ([\d.]+) ms", text)
        acc = re.search(r"agreement with the human label: (\d+)/(\d+)", text)
        route = re.search(r"\n  backend: (\S+)", text)
        ex = re.search(r"ggufExecutor=(\w+)", text)
        thr = re.search(r"ggufThreads=(\d+)", text)
        if not p50: continue
        assert thr and int(thr.group(1)) == threads, f"{log.name}: toggle says {thr and thr.group(1)}"
        out.setdefault(threads, []).append(
            (float(p50.group(1)), acc.group(0).split(": ")[1] if acc else "?",
             route.group(1) if route else "?", ex.group(1) if ex else "?"))
    return out

def table(name, d):
    a = arms(d)
    if not a:
        print(f"\n### {name}: no logs\n"); return
    print(f"\n### {name}\n")
    print("| threads | n | median p50 | min | max | spread | accuracy | route |")
    print("|--------:|--:|-----------:|----:|----:|-------:|:---------|:------|")
    best = min(st.median([x[0] for x in v]) for v in a.values())
    for t in sorted(a):
        v = sorted(x[0] for x in a[t])
        med, lo, hi = st.median(v), v[0], v[-1]
        accs = sorted({x[1] for x in a[t]})
        routes = sorted({f"{x[2]}/{x[3]}" for x in a[t]})
        print(f"| {t} | {len(v)} | **{med:.1f}** | {lo:.1f} | {hi:.1f} | "
              f"{(hi-lo)/med*100:.0f}% | {','.join(accs)} | {','.join(routes)} |")
    print(f"\nBest median: {best:.1f} ms.")
    for t in sorted(a):
        v = [x[0] for x in a[t]]
        print(f"- {t} threads: {st.median(v)/best:.2f}x the best, "
              f"repeats {' '.join(f'{x:.1f}' for x in sorted(v))}")

for name, d in [("rockhopper -- i9-14900HX, hybrid", "raw/rockhopper"),
                ("threadsweep VPS -- AMD EPYC-Milan ccx33, 4 cores x 2 SMT", "raw/vps")]:
    table(name, d)
