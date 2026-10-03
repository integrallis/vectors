#!/usr/bin/env python3
"""Reuse the frozen jars from run.py for an independent persistent-corpus comparison."""

import argparse, hashlib, json, os, shutil, statistics, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("frozen", type=Path)
p.add_argument("generation", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--rounds", type=int, default=3)
a = p.parse_args()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
frozen = a.frozen.resolve()
generation = a.generation.resolve()
java = (
    Path(os.environ["JAVA_HOME"]) / "bin/java"
    if "JAVA_HOME" in os.environ
    else Path(shutil.which("java"))
)
cp = {
    arm: os.pathsep.join(
        str(p) for p in sorted((frozen / (arm + "-jars")).glob("*.jar"))
    )
    for arm in ["baseline", "candidate"]
}
classes = out / "classes"
classes.mkdir()
source = Path(__file__).with_name("CollectionSearchPerformance.java").resolve()
shutil.copy2(source, out / source.name)
subprocess.run(
    [
        str(java.with_name("javac")),
        "--add-modules=jdk.incubator.vector",
        "-cp",
        str(frozen / "classes") + os.pathsep + cp["baseline"],
        "-d",
        str(classes),
        str(source),
    ],
    check=True,
)
provenance = {
    "frozen_runtime": json.loads((frozen / "provenance.json").read_text()),
    "generation": str(generation),
    "sha256": {
        name: hashlib.file_digest((generation / name).open("rb"), "sha256").hexdigest()
        for name in ["vectors.bin", "graph.bin", "manifest.bin"]
    },
}
(out / "provenance.json").write_text(json.dumps(provenance, indent=2))
commands = []
samples = {}
hashes = {}
for r in range(a.rounds):
    for arm in (["baseline", "candidate"] if r % 2 == 0 else ["candidate", "baseline"]):
        name = f"{arm}-{r}"
        command = [
            str(java),
            "--add-modules=jdk.incubator.vector",
            "--enable-native-access=ALL-UNNAMED",
            "-Xmx3g",
            "-cp",
            str(classes) + os.pathsep + str(frozen / "classes") + os.pathsep + cp[arm],
            "CollectionSearchPerformance",
            str(generation),
        ]
        commands.append(command)
        (out / "commands.json").write_text(json.dumps(commands, indent=2))
        print(name, flush=True)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        values = {}
        for line in (out / (name + ".log")).read_text().splitlines():
            if not line.startswith(("flat-", "hnsw-")):
                continue
            key, dim, n, k, round, ns, h = line.split(",")
            values.setdefault(key, []).append(int(ns))
            hashes.setdefault(arm, {}).setdefault(key, set()).add(h)
        for key, ns in values.items():
            samples.setdefault(arm, {}).setdefault(key, []).append(
                statistics.median(ns)
            )
    if hashes["baseline"] != hashes["candidate"]:
        raise SystemExit("result parity failed")
summary = {}
for key in samples["baseline"]:
    b = statistics.median(samples["baseline"][key])
    c = statistics.median(samples["candidate"][key])
    summary[key] = {
        "baseline_ms": b / 1e6,
        "candidate_ms": c / 1e6,
        "less_time_percent": 100 * (1 - c / b),
        "exact_result_parity": True,
        "baseline_forks_ns": samples["baseline"][key],
        "candidate_forks_ns": samples["candidate"][key],
    }
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
