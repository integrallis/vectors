#!/usr/bin/env python3
"""Turns a GgufRowExecutorDispatchBenchmark JMH run into the A/B table.

Reads the raw JMH JSON and derives every number, so the table in a write-up is never typed by
hand. Reports each implementation's score with its error, and the ratio between them at each
thread count -- the ratio is the question, since an implementation that wins at two threads and
loses at twelve has no business being a default either way.
"""
import json
import sys
from collections import defaultdict


def main(path: str) -> int:
    with open(path) as handle:
        rows = json.load(handle)

    # (benchmark, parallelism) -> mode -> (score, error, unit)
    table: dict = defaultdict(dict)
    for row in rows:
        name = row["benchmark"].rsplit(".", 1)[-1]
        params = row["params"]
        mode = params["executorMode"]
        threads = int(params["parallelism"])
        metric = row["primaryMetric"]
        table[(name, threads)][mode] = (
            metric["score"],
            metric.get("scoreError") or float("nan"),
            metric["scoreUnit"],
        )

    benchmarks = sorted({name for name, _ in table})
    for name in benchmarks:
        entries = sorted(t for n, t in table if n == name)
        unit = next(iter(table[(name, entries[0])].values()))[2]
        print(f"\n## {name}  ({unit})\n")
        print(f"{'threads':>7}  {'persistent':>22}  {'countdown':>22}  {'ratio':>7}  verdict")
        print(f"{'-'*7}  {'-'*22}  {'-'*22}  {'-'*7}  -------")
        for threads in entries:
            cell = table[(name, threads)]
            if "persistent" not in cell or "countdown" not in cell:
                print(f"{threads:>7}  incomplete: {sorted(cell)}")
                continue
            p_score, p_err, _ = cell["persistent"]
            c_score, c_err, _ = cell["countdown"]
            ratio = p_score / c_score if c_score else float("inf")
            # Overlapping error bars mean the run cannot tell them apart, which is a result and
            # must not be read as a win for whichever number happens to be lower.
            separated = abs(p_score - c_score) > (
                (0 if p_err != p_err else p_err) + (0 if c_err != c_err else c_err)
            )
            if not separated:
                verdict = "no difference resolved"
            elif ratio > 1:
                verdict = f"countdown faster {ratio:.2f}x"
            else:
                verdict = f"persistent faster {1 / ratio:.2f}x"
            print(
                f"{threads:>7}  {p_score:>12.3f} +- {p_err:<7.3f}  "
                f"{c_score:>12.3f} +- {c_err:<7.3f}  {ratio:>7.2f}  {verdict}"
            )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print("usage: summarize-dispatch-ab.py <jmh-result.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
