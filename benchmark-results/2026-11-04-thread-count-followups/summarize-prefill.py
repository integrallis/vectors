#!/usr/bin/env python3
"""Derive the prefill/decode table from raw driver logs. Nothing here is hand-typed."""
import re, sys, statistics as st
from pathlib import Path

rows = {}
for log in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "raw/rockhopper-pd").glob("pd-*.log")):
    threads = int(re.match(r"pd-(\d+)\.log", log.name).group(1))
    text = log.read_text(errors="replace")
    toggle = re.search(r"threads toggle: (\d+)", text)
    assert toggle and int(toggle.group(1)) == threads, f"{log.name}: toggle {toggle and toggle.group(1)}"
    for line in text.splitlines():
        if not line.startswith("rep "):
            continue
        pre = re.search(r"prefill=([\d.]+) ms \(([\d.]+) tok/s\)", line)
        dec = re.search(r"decode=([\d.]+) ms \(([\d.]+) tok/s\)", line)
        cache = re.search(r"cacheReadInputTokens=(\d+)", line)
        ptok = re.search(r"promptTokens=(\d+)", line)
        assert cache and int(cache.group(1)) <= 16, (
            f"{log.name}: prompt cache served {cache.group(1)} tokens -- prefill not measured")
        rows.setdefault(threads, []).append(
            (float(pre.group(2)), float(dec.group(2)), int(ptok.group(1))))

if not rows:
    print("no logs"); raise SystemExit(0)

print("| threads | n | prompt tokens | prefill tok/s (median) | range | decode tok/s (median) | range |")
print("|--------:|--:|--------------:|-----------------------:|:------|----------------------:|:------|")
for t in sorted(rows):
    v = rows[t]
    pre = sorted(x[0] for x in v); dec = sorted(x[1] for x in v)
    toks = sorted({x[2] for x in v})
    print(f"| {t} | {len(v)} | {','.join(map(str,toks))} | **{st.median(pre):.1f}** | "
          f"{pre[0]:.1f}-{pre[-1]:.1f} | **{st.median(dec):.2f}** | {dec[0]:.2f}-{dec[-1]:.2f} |")

bp = max(st.median([x[0] for x in v]) for v in rows.values())
bd = max(st.median([x[1] for x in v]) for v in rows.values())
print(f"\nBest prefill {bp:.1f} tok/s, best decode {bd:.2f} tok/s.")
for t in sorted(rows):
    v = rows[t]
    print(f"- {t} threads: prefill {st.median([x[0] for x in v])/bp:.2f}x of best, "
          f"decode {st.median([x[1] for x in v])/bd:.2f}x of best")
