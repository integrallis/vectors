#!/usr/bin/env python3
"""Alternate frozen runtimes on an idle host; retain every command/sample and exact output digest."""

import argparse, hashlib, json, os, platform, statistics, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("inputs", type=Path)
p.add_argument("collection", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--query-counts", default="1,2,3,4,16,64")
p.add_argument("--public-api", action="store_true")
p.add_argument("--baseline-sha", required=True)
p.add_argument("--candidate-sha", required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
java = Path(os.environ["JAVA_HOME"]) / "bin/java"
source = Path(__file__).with_name(
    "CollectionBatchPerformance.java" if a.public_api else "FlatBatchPerformance.java"
)
sources = [source] + (
    [Path(__file__).with_name("FlatBatchPerformance.java")] if a.public_api else []
)
classes = a.output / "classes"
classes.mkdir()
subprocess.run(
    [
        str(java.with_name("javac")),
        "--add-modules=jdk.incubator.vector",
        "-cp",
        str(a.baseline / "*"),
        "-d",
        str(classes),
        *[str(f) for f in sources],
    ],
    check=True,
)
provenance = {
    "host": platform.platform(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "java": subprocess.check_output(
        [str(java), "-version"], stderr=subprocess.STDOUT, text=True
    ),
    "baseline_sha": a.baseline_sha,
    "candidate_sha": a.candidate_sha,
    "harness_sha256": {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sources
    },
    "api": "VectorCollection" if a.public_api else "FlatScanAdapter",
    "jars": {
        arm: {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in getattr(a, arm).glob("*.jar")
        }
        for arm in ("baseline", "candidate")
    },
    "inputs": {
        str(f): hashlib.sha256(f.read_bytes()).hexdigest()
        for root in (a.inputs, a.collection)
        for f in root.iterdir()
        if f.is_file()
    },
}
(a.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
cases = [
    [
        "glove",
        str(a.inputs / "glove-100-angular-train.fbin"),
        str(a.inputs / "glove-100-angular-test.fbin"),
        "COSINE",
    ],
    ["collection", str(a.collection), "-", "COSINE"],
    [
        "fashion-control",
        str(a.inputs / "fashion-mnist-784-euclidean-train.fbin"),
        str(a.inputs / "fashion-mnist-784-euclidean-test.fbin"),
        "EUCLIDEAN",
    ],
]
commands = []
samples = {}
digests = {}
for repeat in range(a.rounds):
    for case in cases:
        for arm in (
            ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        ):
            name = f"{arm}-{repeat}-{case[0]}"
            print(name, flush=True)
            cmd = [
                str(java),
                "--add-modules=jdk.incubator.vector",
                "--enable-native-access=ALL-UNNAMED",
                "-Xmx4g",
                "-Xbatch",
                "-cp",
                str(classes) + os.pathsep + str(getattr(a, arm) / "*"),
                source.stem,
                *case,
                a.query_counts,
                "8",
                "7",
            ]
            commands.append(cmd)
            (a.output / "commands.json").write_text(json.dumps(commands, indent=2))
            log = a.output / (name + ".log")
            with log.open("w") as out:
                subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True)
            current = {}
            for line in log.read_text().splitlines():
                if not line.startswith("FLAT,"):
                    continue
                _, corpus, mode, n, d, q, sample, ns, digest = line.split(",")
                key = f"{corpus}-{mode}-n{n}-d{d}-q{q}"
                current.setdefault(key, []).append(int(ns))
                digests.setdefault((arm, key), set()).add(digest)
            if len(current) != 2 * len(a.query_counts.split(",")) or any(
                len(times) != 7 for times in current.values()
            ):
                raise SystemExit("Incomplete flat samples: " + name)
            for key, times in current.items():
                samples.setdefault((arm, key), []).append(statistics.median(times))
        for key in current:
            if digests["baseline", key] != digests["candidate", key]:
                raise SystemExit("EXACT PARITY FAILED: " + key)
summary = {}
for arm, key in samples:
    if arm != "baseline":
        continue
    b = samples["baseline", key]
    c = samples["candidate", key]
    summary[key] = {
        "baseline_medians_ns": b,
        "candidate_medians_ns": c,
        "median_time_reduction_pct": 100
        * (1 - statistics.median(c) / statistics.median(b)),
        "exact_result_parity": digests["baseline", key] == digests["candidate", key],
    }
(a.output / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
