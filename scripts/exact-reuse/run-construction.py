#!/usr/bin/env python3
"""Paired construction ablation. Exact single-worker graphs; unchanged multithread recall gates."""

import argparse, hashlib, json, os, platform, statistics, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("inputs", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--rows", type=int, default=20000)
p.add_argument("--seeds", default="17,42,91")
p.add_argument("--threads", default="1,8")
p.add_argument("--baseline-sha", required=True)
p.add_argument("--candidate-sha", required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
java = Path(os.environ["JAVA_HOME"]) / "bin/java"
sources = [
    Path(__file__).with_name(n)
    for n in ("FlatBatchPerformance.java", "ConstructionReusePerformance.java")
]
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
    "sources": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
    "jars": {
        arm: {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in getattr(a, arm).glob("*.jar")
        }
        for arm in ("baseline", "candidate")
    },
    "inputs": {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in a.inputs.glob("*.fbin")
    },
    "rows": a.rows,
    "M": 16,
    "efConstruction": 200,
    "k": 10,
    "recall_gate": -0.005,
    "seeds": a.seeds,
    "threads": a.threads,
}
(a.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
commands = []
records = []
for corpus, metric in [
    ("glove-100-angular", "COSINE"),
    ("fashion-mnist-784-euclidean", "EUCLIDEAN"),
]:
    for threads in map(int, a.threads.split(",")):
        for repeat, seed in enumerate(map(int, a.seeds.split(","))):
            pair = {}
            for arm in (
                ("baseline", "candidate")
                if repeat % 2 == 0
                else ("candidate", "baseline")
            ):
                name = f"{corpus}-t{threads}-s{seed}-{arm}"
                print(name, flush=True)
                oracle = a.output / f"{corpus}-oracle.bin"
                if not oracle.exists() and arm != "baseline":
                    raise SystemExit("Oracle must be frozen by baseline first")
                cmd = [
                    str(java),
                    "--add-modules=jdk.incubator.vector",
                    "--enable-native-access=ALL-UNNAMED",
                    "-Xmx6g",
                    "-Xbatch",
                    "-cp",
                    str(classes) + os.pathsep + str(getattr(a, arm) / "*"),
                    "ConstructionReusePerformance",
                    str(a.inputs / f"{corpus}-train.fbin"),
                    str(a.inputs / f"{corpus}-test.fbin"),
                    str(a.rows),
                    str(threads),
                    str(seed),
                    metric,
                    str(oracle),
                    "2048",
                    "7",
                ]
                commands.append(cmd)
                (a.output / "commands.json").write_text(json.dumps(commands, indent=2))
                log = a.output / (name + ".log")
                with log.open("w") as out:
                    subprocess.run(
                        cmd, stdout=out, stderr=subprocess.STDOUT, check=True
                    )
                r = {
                    "corpus": corpus,
                    "threads": threads,
                    "seed": seed,
                    "arm": arm,
                    "recall": {},
                    "search_ns": {},
                    "search_hash": {},
                }
                for line in log.read_text().splitlines():
                    f = line.split(",")
                    if f[0] == "BUILD":
                        r.update(build_ns=int(f[5]), graph_hash=f[6])
                    elif f[0] == "RECALL":
                        r["recall"][f[1]] = float(f[4])
                    elif f[0] == "SEARCH":
                        r["search_ns"].setdefault(f[1], []).append(int(f[3]))
                        r["search_hash"][f[1]] = f[4]
                if (
                    "build_ns" not in r
                    or set(r["recall"]) != {"32", "128", "512"}
                    or any(
                        len(r["search_ns"].get(ef, [])) != 7
                        for ef in ("32", "128", "512")
                    )
                ):
                    raise SystemExit("Incomplete construction samples: " + name)
                records.append(r)
                pair[arm] = r
                (a.output / "records.json").write_text(json.dumps(records, indent=2))
            b, c = pair["baseline"], pair["candidate"]
            parity = (
                b["graph_hash"] == c["graph_hash"]
                and b["search_hash"] == c["search_hash"]
            )
            gate = all(
                c["recall"][ef] - value >= -0.005 for ef, value in b["recall"].items()
            )
            if threads == 1 and not parity:
                raise SystemExit("Single-worker exact graph/result parity failed")
            if not gate:
                print("RECALL GATE FAILED: " + name, flush=True)
summary = {}
for corpus in sorted({r["corpus"] for r in records}):
    for threads in sorted({r["threads"] for r in records}):
        rows = [r for r in records if r["corpus"] == corpus and r["threads"] == threads]
        b = [r for r in rows if r["arm"] == "baseline"]
        c = [r for r in rows if r["arm"] == "candidate"]
        delta = {
            ef: [y["recall"][ef] - x["recall"][ef] for x, y in zip(b, c)]
            for ef in b[0]["recall"]
        }
        summary[f"{corpus}-t{threads}"] = {
            "baseline_build_ns": [r["build_ns"] for r in b],
            "candidate_build_ns": [r["build_ns"] for r in c],
            "median_build_time_reduction_pct": 100
            * (
                1
                - statistics.median(r["build_ns"] for r in c)
                / statistics.median(r["build_ns"] for r in b)
            ),
            "recall_delta": delta,
            "recall_gate_pass": all(d >= -0.005 for ds in delta.values() for d in ds),
            "graph_parity": [x["graph_hash"] == y["graph_hash"] for x, y in zip(b, c)],
        }
(a.output / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
