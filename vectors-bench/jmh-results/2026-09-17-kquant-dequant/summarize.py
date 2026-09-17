#!/usr/bin/env python3
"""Summarize the SIMD K-quant dequant runs into Markdown and evaluate the pre-registered rule.

Usage: summarize.py raw/<label> [--candidate simd-int]

Reads dequant.json and ab[-st]-{integer,band-<arm>}[-r<N>].json written by run.sh. Rounds are
reported separately (never averaged), so a reader can see whether an effect repeats.

Rule parts evaluated per run (see README.md):
  (b) candidate dequant throughput >= 2.0x scalar for Q4_K and Q6_K on both shapes;
  (c) BAND_F32 with candidate vs scalar: batch 1 and 4 mean ms/op lower in every cell, batch 512
      mean ms/op <= 1.03x scalar in every cell.
Without --candidate the candidate is this run's best geometric-mean dequant arm; the pre-registered
candidate comes from the Genoa 256-bit run, so pass it explicitly for every other run.
"""
import glob, json, math, os, re, sys

SHAPES = ["8192x2560", "2560x8192"]
FORMATS = ["Q4_K", "Q6_K"]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def dequant_table(d):
    rows = {}
    for r in load(os.path.join(d, "dequant.json")):
        p = r["params"]
        rows[(p["format"], p["shape"], p["dequant"])] = (r["primaryMetric"]["score"], r["primaryMetric"]["scoreError"])
    if not rows:
        return {}, None
    arms = sorted({k[2] for k in rows}, key=lambda a: (a != "scalar", a))
    print("\n### Dequantisation alone (single thread; ms/op, M F32 elements/s, speedup vs scalar)\n")
    print("| format | shape | " + " | ".join(arms) + " |")
    print("|---|---|" + "---:|" * len(arms))
    speed = {}
    for fmt in FORMATS:
        for shape in SHAPES:
            m, k = map(int, shape.split("x"))
            cells = []
            base = rows.get((fmt, shape, "scalar"))
            for arm in arms:
                v = rows.get((fmt, shape, arm))
                if not v:
                    cells.append("—")
                    continue
                ms, err = v
                meps = m * k / (ms / 1000.0) / 1e6
                s = base[0] / ms if base else float("nan")
                speed[(fmt, shape, arm)] = s
                cells.append(f"{ms:.2f} ± {err:.2f} / {meps:.0f} M / **{s:.2f}x**")
            print(f"| {fmt} | {shape} | " + " | ".join(cells) + " |")
    geo = {}
    for arm in arms:
        vals = [speed[(f, s, arm)] for f in FORMATS for s in SHAPES if (f, s, arm) in speed]
        if len(vals) == 4:
            geo[arm] = math.exp(sum(math.log(v) for v in vals) / 4)
    print("\nGeometric-mean dequant speedup vs scalar: " + ", ".join(f"{a} {g:.2f}x" for a, g in geo.items()))
    best = max((a for a in geo if a != "scalar"), key=lambda a: geo[a], default=None)
    return speed, best


def ab_runs(d, suffix):
    """{round_tag: {(kernel_or_arm): {(fmt, shape, batch): (ms, err)}}}"""
    runs = {}
    pat = re.compile(rf"^ab{re.escape(suffix)}-(integer|band-(.+?))(-r\d+)?\.json$")
    for path in sorted(glob.glob(os.path.join(d, f"ab{suffix}-*.json"))):
        m = pat.match(os.path.basename(path))
        if not m:
            continue
        arm = "integer" if m.group(1) == "integer" else m.group(2)
        tag = m.group(3) or ""
        for r in load(path):
            p = r["params"]
            key = (p["format"], p["shape"], int(p["batchSize"]))
            runs.setdefault(tag, {}).setdefault(arm, {})[key] = (r["primaryMetric"]["score"], r["primaryMetric"]["scoreError"])
    return runs


def ab_table(title, arms_rows, candidate):
    arms = [a for a in ["scalar", "simd-byte", "simd-int", "simd-split"] if a in arms_rows]
    if "scalar" not in arms_rows:
        return None
    print(f"\n### {title}\n")
    print("ms/op ± 99.9% half-width. `new/old` = scalar band ms / arm band ms (>1 = faster than scalar dequant). `vs int` = integer ms / band ms.\n")
    cols = ["format", "shape", "batch", "integer"] + [f"band {a}" for a in arms]
    cols += [f"{a} new/old" for a in arms if a != "scalar"] + [f"{a} vs int" for a in arms]
    print("| " + " | ".join(cols) + " |")
    print("|---|---|" + "---:|" * (len(cols) - 2))
    keys = sorted(arms_rows["scalar"], key=lambda k: (k[0], k[1], k[2]))
    verdict = {"b1b4": [], "b512": []}
    for key in keys:
        fmt, shape, batch = key
        integer = arms_rows.get("integer", {}).get(key)
        cells = [fmt, shape, str(batch), f"{integer[0]:.2f} ± {integer[1]:.2f}" if integer else "—"]
        for a in arms:
            v = arms_rows[a].get(key)
            cells.append(f"{v[0]:.2f} ± {v[1]:.2f}" if v else "—")
        old = arms_rows["scalar"][key][0]
        for a in arms:
            if a == "scalar":
                continue
            v = arms_rows[a].get(key)
            cells.append(f"**{old / v[0]:.2f}x**" if v else "—")
        for a in arms:
            v = arms_rows[a].get(key)
            cells.append(f"{integer[0] / v[0]:.2f}x" if (v and integer) else "—")
        print("| " + " | ".join(cells) + " |")
        if candidate and candidate in arms_rows and key in arms_rows[candidate]:
            new = arms_rows[candidate][key][0]
            if batch in (1, 4):
                verdict["b1b4"].append((key, new < old, old / new))
            if batch == 512:
                verdict["b512"].append((key, new <= 1.03 * old, new / old))
    return verdict


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    d = sys.argv[1]
    candidate = None
    if "--candidate" in sys.argv:
        candidate = sys.argv[sys.argv.index("--candidate") + 1]
    host = os.path.join(d, "host.txt")
    if os.path.exists(host):
        print("```\n" + open(host).read().strip() + "\n```")
    speed, best = dequant_table(d)
    if candidate is None:
        candidate = best
        print(f"\nCandidate (best geomean in THIS run; the rule fixes it from Genoa 256-bit): {candidate}")
    else:
        print(f"\nCandidate (given): {candidate}")
    if speed and candidate:
        cells = [(f, s, speed.get((f, s, candidate))) for f in FORMATS for s in SHAPES]
        ok = all(v is not None and v >= 2.0 for _, _, v in cells)
        print(f"\n**(b) dequant >= 2.0x scalar, all 4 cells: {'PASS' if ok else 'FAIL'}** — "
              + ", ".join(f"{f} {s} {v:.2f}x" for f, s, v in cells if v is not None))
    for suffix, label in (("", "multi-threaded (executor default)"), ("-st", "single-threaded")):
        for tag, arms_rows in sorted(ab_runs(d, suffix).items()):
            verdict = ab_table(f"Band A/B, {label}{' round ' + tag[2:] if tag else ''}", arms_rows, candidate)
            if verdict and suffix == "" and candidate and not verdict["b1b4"]:
                print(f"\n(c) {candidate}: no data in this round yet")
            elif verdict and suffix == "" and candidate:
                ok1 = bool(verdict["b1b4"]) and all(x[1] for x in verdict["b1b4"])
                ok2 = bool(verdict["b512"]) and all(x[1] for x in verdict["b512"])
                print(f"\n**(c) {candidate}{' round ' + tag[2:] if tag else ''}: batch 1/4 faster than scalar in every cell: "
                      f"{'PASS' if ok1 else 'FAIL'} ({sum(x[1] for x in verdict['b1b4'])}/{len(verdict['b1b4'])}); "
                      f"batch 512 <= 1.03x scalar in every cell: {'PASS' if ok2 else 'FAIL'} "
                      f"(worst {max((x[2] for x in verdict['b512']), default=float('nan')):.3f}x)**")


if __name__ == "__main__":
    main()
