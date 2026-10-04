#!/usr/bin/env python3
"""Independently recompute the three-pair flat qualification from its raw logs."""

import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
reported = json.loads((root / "summary.json").read_text())
data, digests = {}, {}
samples = 0
for file in root.glob("*.log"):
    parts = file.stem.split("-", 2)
    if len(parts) != 3 or parts[0] not in {"baseline", "candidate"}:
        continue
    arm, fork, _ = parts
    seen = {}
    for line in file.read_text().splitlines():
        if not line.startswith("FLAT,"):
            continue
        _, corpus, mode, n, d, q, sample, ns, digest = line.split(",")
        key = f"{corpus}-{mode}-n{n}-d{d}-q{q}"
        assert int(sample) not in seen.setdefault(key, {}), (file, key, sample)
        seen[key][int(sample)] = int(ns)
        digests.setdefault(key, set()).add(digest)
        samples += 1
    assert seen, file
    for key, values in seen.items():
        assert set(values) == set(range(7)), (file, key)
        data.setdefault(key, {}).setdefault(arm, {})[int(fork)] = statistics.median(
            values.values()
        )

assert set(data) == set(reported)
assert samples == len(data) * 2 * 3 * 7
for key, arms in data.items():
    assert set(arms) == {"baseline", "candidate"}
    assert len(digests[key]) == 1, (key, "digest mismatch")
    assert set(arms["baseline"]) == set(arms["candidate"]) == {0, 1, 2}
    baseline = [value for _, value in sorted(arms["baseline"].items())]
    candidate = [value for _, value in sorted(arms["candidate"].items())]
    result = reported[key]
    assert baseline == result["baseline_medians_ns"]
    assert candidate == result["candidate_medians_ns"]
    assert math.isclose(
        100 * (1 - statistics.median(candidate) / statistics.median(baseline)),
        result["median_time_reduction_pct"],
        abs_tol=1e-9,
    )
    assert result["exact_result_parity"]

gates = {
    key: value["median_time_reduction_pct"]
    for key, value in reported.items()
    if "-batch-" in key and key.endswith("-q64") and not key.startswith("fashion")
}
assert len(gates) == 2 and all(value >= 5 for value in gates.values()), gates
result = {
    "source": "independent raw FLAT log parser",
    "cases": len(data),
    "raw_measurements": samples,
    "forks_per_arm": 3,
    "samples_per_fork": 7,
    "all_raw_digest_sets_identical": True,
    "reported_medians_recomputed": True,
    "q64_gates": gates,
}
(root / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
