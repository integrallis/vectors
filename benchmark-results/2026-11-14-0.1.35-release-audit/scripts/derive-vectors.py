#!/usr/bin/env python3
"""Derive the vectors 0.1.34 vs 0.1.35 comparison from the box's raw artifacts.

Inputs (as the driver wrote them):
  out/jmh/<tag>-<bench>-r<round>.json   JMH JSON, fork=2 warmup=3 iterations=5, interleaved per benchmark
  out/recall/<tag>-record.log           recallGate in record mode: one "<INDEX> <params> <recall>" row per line
  out/recall/v0.1.35-verify.log         recallGate in verify mode against the committed (k-means era) baseline
  out/slow/<tag>/*.xml                  SIFT-small HNSW/Vamana recall floor tests

Every figure printed is read from a file; none is typed. JMH ratios are median(v0.1.35) / median(v0.1.34)
of the primary score across all forks and rounds; the unit is JMH's own and is printed with it.

    python3 derive-vectors.py <out-dir>
"""
import json, pathlib, re, statistics, sys
import xml.etree.ElementTree as ET

TAGS = ["v0.1.34", "v0.1.35"]


def jmh_scores(path):
    """Primary score of every fork of every benchmark method in one JMH JSON file."""
    out = {}
    for record in json.loads(path.read_text()):
        name = record["benchmark"].split(".")[-2] + "." + record["benchmark"].split(".")[-1]
        params = record.get("params") or {}
        key = name + (("{" + ",".join(f"{k}={v}" for k, v in sorted(params.items())) + "}") if params else "")
        mode = record["mode"]; unit = record["primaryMetric"]["scoreUnit"]
        scores = [s for fork in record["primaryMetric"]["rawData"] for s in fork]
        out.setdefault(key, {"mode": mode, "unit": unit, "scores": []})["scores"] += scores
    return out


def main():
    root = pathlib.Path(sys.argv[1])
    print("host:", " / ".join(l.split(":", 1)[1].strip() for l in (root / "cpu.txt").read_text().splitlines()
                            if l.startswith(("Model name", "Core(s) per socket", "Thread(s) per core"))))
    print("java:", (root / "java-version.txt").read_text().splitlines()[0])
    print()
    # ---- JMH ----
    pooled = {t: {} for t in TAGS}
    files = {t: sorted(f for d in ("jmh", "jmh2") for f in (root / d).glob(f"{t}-*.json")) for t in TAGS}
    print("JMH files:", {t: len(v) for t, v in files.items()})
    for t in TAGS:
        for f in files[t]:
            for key, rec in jmh_scores(f).items():
                p = pooled[t].setdefault(key, {"mode": rec["mode"], "unit": rec["unit"], "scores": []})
                p["scores"] += rec["scores"]
    keys = sorted(set(pooled["v0.1.34"]) & set(pooled["v0.1.35"]))
    print()
    print("| benchmark | mode | v0.1.34 median [min..max] | v0.1.35 median [min..max] | 0.1.35 / 0.1.34 | reading |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key in keys:
        a, b = pooled["v0.1.34"][key], pooled["v0.1.35"][key]
        ma, mb = statistics.median(a["scores"]), statistics.median(b["scores"])
        ratio = mb / ma if ma else float("nan")
        # thrpt: higher is better; avgt/sample/ss: lower is better
        better_high = a["mode"] == "thrpt"
        change = (ratio - 1) if better_high else (1 - ratio)
        reading = ("faster" if change > 0 else "slower") + f" by {abs(change) * 100:.1f}%"
        print(f"| {key} | {a['mode']} ({a['unit']}) | {ma:.3f} [{min(a['scores']):.3f}..{max(a['scores']):.3f}] (n={len(a['scores'])}) "
              f"| {mb:.3f} [{min(b['scores']):.3f}..{max(b['scores']):.3f}] (n={len(b['scores'])}) | {ratio:.3f} | {reading} |")
    missing = {t: sorted(set(pooled[TAGS[1 - i]]) - set(pooled[t])) for i, t in enumerate(TAGS)}
    for t, m in missing.items():
        if m: print(f"(only in the other tag, not compared: {m})")
    # ---- recall: record mode on both tags ----
    print()
    rows = {}
    for t in TAGS:
        log = root / "recall" / f"{t}-record.log"
        if not log.is_file(): continue
        for line in log.read_text().splitlines():
            m = re.match(r"^\s*(HNSW|VAMANA|IVF_FLAT|IVF_PQ)\s+(\S+)\s+([0-9.]+)\s*$", line)
            if m: rows.setdefault((m.group(1), m.group(2)), {})[t] = float(m.group(3))
    baseline = {}
    for line in (root / ".." / "recall-baseline.txt").read_text().splitlines() if (root / ".." / "recall-baseline.txt").is_file() else []:
        m = re.match(r"^\s*(HNSW|VAMANA|IVF_FLAT|IVF_PQ)\s+(\S+)\s+([0-9.]+)", line)
        if m: baseline[(m.group(1), m.group(2))] = float(m.group(3))
    print("| recall@10 row | committed baseline (2026-06, k-means) | v0.1.34 record | v0.1.35 record | 0.1.35 − 0.1.34 |")
    print("| --- | --- | --- | --- | --- |")
    for key in sorted(rows):
        a, b = rows[key].get("v0.1.34"), rows[key].get("v0.1.35")
        base = baseline.get(key)
        d = (b - a) if (a is not None and b is not None) else None
        print(f"| {key[0]} {key[1]} | {base if base is not None else 'n/a'} | {a if a is not None else 'n/a'} | {b if b is not None else 'n/a'} | {('%+.4f' % d) if d is not None else 'n/a'} |")
    # ---- recall: verify mode on v0.1.35 ----
    v = root / "recall" / "v0.1.35-verify.log"
    if v.is_file():
        text = v.read_text()
        verdict = "PASSED" if "BUILD SUCCESSFUL" in text else ("FAILED" if "BUILD FAILED" in text else "unknown")
        print(f"\nrecallGate verify on v0.1.35 against the committed baseline: {verdict}")
        for line in text.splitlines():
            if re.search(r"recall|drop|tolerance|baseline", line, re.I) and not line.startswith(">"): print("   ", line.strip()[:160])
    # ---- SIFT-small floors ----
    print()
    for t in TAGS:
        tot = fail = 0; names = []
        for x in (root / "slow" / t).glob("*.xml"):
            r = ET.parse(x).getroot(); tot += int(r.get("tests", 0)); fail += int(r.get("failures", 0)) + int(r.get("errors", 0))
            names += [tc.get("name") for tc in r.iter("testcase")]
        print(f"SIFT-small recall floors {t}: tests={tot} failures={fail}  {sorted(names)}")


if __name__ == "__main__":
    main()
