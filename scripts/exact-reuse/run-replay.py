#!/usr/bin/env python3
"""Replay a saved collection through ordinary commits, then compare reopened recall."""

import argparse, hashlib, json, os, platform, statistics, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
for name in ("baseline", "candidate", "source", "generation", "output"):
    p.add_argument(name, type=Path)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--count", type=int, default=100000)
p.add_argument("--cadence", type=int, default=25000)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--queries", type=int, default=1000)
p.add_argument(
    "--recall-only",
    action="store_true",
    help="Resume recall for completed timing records",
)
p.add_argument("--baseline-sha", required=True)
p.add_argument("--candidate-sha", required=True)
a = p.parse_args()
for name in ("baseline", "candidate", "source", "generation", "output"):
    setattr(a, name, getattr(a, name).resolve())
a.output.mkdir(parents=True, exist_ok=a.recall_only)
java = Path(os.environ["JAVA_HOME"]) / "bin/java"
classes = a.output / "classes"
classes.mkdir(exist_ok=a.recall_only)
sources = [
    a.source
    / "vectors-bench/src/main/java/com/integrallis/vectors/bench"
    / f"{name}.java"
    for name in ("DefaultCollectionReplayBenchmark", "CollectionRecallProbe")
]
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
    "args": {k: str(v) for k, v in vars(a).items()},
    "host": platform.platform(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "java": subprocess.check_output(
        [str(java), "-version"], stderr=subprocess.STDOUT, text=True
    ),
    "source_sha256": {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sources
    },
    "inputs_sha256": {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in a.generation.iterdir()
        if f.is_file()
    },
    "jars": {
        arm: {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in getattr(a, arm).glob("*.jar")
        }
        for arm in ("baseline", "candidate")
    },
}
(
    a.output / ("recall-resume-provenance.json" if a.recall_only else "provenance.json")
).write_text(json.dumps(provenance, indent=2))
commands = json.loads((a.output / "commands.json").read_text()) if a.recall_only else []
records = json.loads((a.output / "records.json").read_text()) if a.recall_only else []


def run(arm, label, main, args):
    cmd = [
        str(java),
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-Xmx6g",
        "-cp",
        str(classes) + os.pathsep + str(getattr(a, arm) / "*"),
        "com.integrallis.vectors.bench." + main,
        *map(str, args),
    ]
    commands.append(cmd)
    (a.output / "commands.json").write_text(json.dumps(commands, indent=2))
    print(label, flush=True)
    log = a.output / (label + ".log")
    if log.exists():
        attempt = 1
        while log.with_suffix(f".attempt{attempt}.log").exists():
            attempt += 1
        log.rename(log.with_suffix(f".attempt{attempt}.log"))
    with log.open("w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True)
    return log.read_text()


for repeat in range(0 if a.recall_only else a.rounds):
    for arm in (
        ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
    ):
        label = f"{arm}-{repeat}"
        destination = a.output / (label + "-collection")
        log = run(
            arm,
            label,
            "DefaultCollectionReplayBenchmark",
            [a.generation, destination, a.cadence, a.threads, a.count],
        )
        value = next(
            line.split(",") for line in log.splitlines() if line.startswith("default,")
        )
        records.append(
            {
                "arm": arm,
                "round": repeat,
                "documents": int(value[1]),
                "commits": int(value[2]),
                "ingest_ms": float(value[3]),
                "reopen_ms": float(value[4]),
            }
        )
        (a.output / "records.json").write_text(json.dumps(records, indent=2))
# Recall runs only after every timing run. Both collections use the same baseline query runtime.
for r in records:
    label = f'{r["arm"]}-{r["round"]}'
    log = run(
        "baseline",
        label + "-recall",
        "CollectionRecallProbe",
        [a.generation, a.output / (label + "-collection"), a.count, a.queries],
    )
    r["recall"] = {
        f[0]: float(f[2])
        for line in log.splitlines()
        if len(f := line.split(",")) == 3
        and f[0] in ("10", "16", "32", "64", "128", "256")
    }
    if set(r["recall"]) != {"10", "16", "32", "64", "128", "256"}:
        raise SystemExit("Incomplete replay recall: " + label)
    (a.output / "records.json").write_text(json.dumps(records, indent=2))
b = sorted((r for r in records if r["arm"] == "baseline"), key=lambda r: r["round"])
c = sorted((r for r in records if r["arm"] == "candidate"), key=lambda r: r["round"])
delta = {
    ef: [y["recall"][ef] - x["recall"][ef] for x, y in zip(b, c)]
    for ef in b[0]["recall"]
}
summary = {
    "baseline_ingest_ms": [r["ingest_ms"] for r in b],
    "candidate_ingest_ms": [r["ingest_ms"] for r in c],
    "median_time_reduction_pct": 100
    * (
        1
        - statistics.median(r["ingest_ms"] for r in c)
        / statistics.median(r["ingest_ms"] for r in b)
    ),
    "recall_delta": delta,
    "recall_gate_pass": all(d >= -0.005 for ds in delta.values() for d in ds),
    "source_unchanged": all(
        hashlib.sha256((a.generation / name).read_bytes()).hexdigest() == digest
        for name, digest in provenance["inputs_sha256"].items()
    ),
}
(a.output / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
