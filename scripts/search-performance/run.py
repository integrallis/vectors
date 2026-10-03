#!/usr/bin/env python3
"""Build/freeze revisions, then alternate fresh JVMs; output hashes must match exactly."""

import argparse, hashlib, json, os, platform, shutil, statistics, subprocess, tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--rounds", type=int, default=3)
a = p.parse_args()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
roots = {k: getattr(a, k).resolve() for k in ("baseline", "candidate")}
java = (
    Path(os.environ["JAVA_HOME"]) / "bin/java"
    if "JAVA_HOME" in os.environ
    else Path(shutil.which("java"))
)
source = Path(__file__).with_name("SearchPerformance.java").resolve()
shutil.copy2(source, out / source.name)
provenance = {
    "machine": platform.platform(),
    "java": subprocess.check_output(
        [str(java), "-version"], stderr=subprocess.STDOUT, text=True
    ),
    "harness_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
cp = {}
with tempfile.TemporaryDirectory() as tmp:
    init = Path(tmp) / "classpath.gradle"
    init.write_text(
        """allprojects { plugins.withId('java') { tasks.register('perfClasspath') { dependsOn tasks.named('classes'), sourceSets.main.runtimeClasspath; doLast { println('RUNTIME_CP=' + sourceSets.main.runtimeClasspath.asPath) } } } }"""
    )
    for arm, root in roots.items():
        provenance[arm] = {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
        }
        (out / (arm + ".patch")).write_bytes(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        )
        log = subprocess.check_output(
            [
                str(root / "gradlew"),
                "-I",
                str(init),
                ":vectors-bench:perfClasspath",
                "--max-workers=2",
                "--console=plain",
            ],
            cwd=root,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (out / (arm + "-build.log")).write_text(log)
        paths = next(
            line.split("=", 1)[1]
            for line in log.splitlines()
            if line.startswith("RUNTIME_CP=")
        )
        frozen = out / (arm + "-jars")
        frozen.mkdir()
        entries = []
        for item in paths.split(os.pathsep):
            path = Path(item)
            if path.is_file() and path.suffix == ".jar":
                target = frozen / path.name
                shutil.copy2(path, target)
                entries.append(str(target))
        cp[arm] = os.pathsep.join(entries)
    classes = out / "classes"
    classes.mkdir()
    subprocess.run(
        [
            str(java.with_name("javac")),
            "--add-modules=jdk.incubator.vector",
            "-cp",
            cp["baseline"],
            "-d",
            str(classes),
            str(source),
        ],
        check=True,
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    commands = []
    for repeat in range(a.rounds):
        for arm in (
            ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        ):
            name = f"{arm}-{repeat}"
            work = out / (name + "-data")
            command = [
                str(java),
                "--add-modules=jdk.incubator.vector",
                "--enable-native-access=ALL-UNNAMED",
                "-Xmx3g",
                "-cp",
                str(classes) + os.pathsep + cp[arm],
                "SearchPerformance",
                str(work),
            ]
            commands.append(command)
            (out / "commands.json").write_text(json.dumps(commands, indent=2))
            print(name, flush=True)
            with (out / (name + ".log")).open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            if all((out / f"{side}-{repeat}.log").exists() for side in roots):
                checks = {}
                for side in roots:
                    checks[side] = {}
                    for line in (out / f"{side}-{repeat}.log").read_text().splitlines():
                        if line.startswith(("flat-", "hnsw-")):
                            fields = line.split(",")
                            checks[side].setdefault(tuple(fields[:4]), set()).add(
                                fields[-1]
                            )
                if checks["baseline"] != checks["candidate"]:
                    (out / f"parity-failure-{repeat}.json").write_text(
                        json.dumps(
                            {
                                side: {str(k): sorted(v) for k, v in rows.items()}
                                for side, rows in checks.items()
                            },
                            indent=2,
                        )
                    )
                    raise SystemExit(
                        "Result parity failed; stopped before further timing runs"
                    )
values = {}
hashes = {}
allocations = {}
for arm in roots:
    values[arm] = {}
    hashes[arm] = {}
    for repeat in range(a.rounds):
        samples = {}
        for line in (out / f"{arm}-{repeat}.log").read_text().splitlines():
            if line.startswith("SEARCHER_ALLOCATION_BYTES="):
                allocations.setdefault(arm, []).append(int(line.split("=")[1]))
            if not line.startswith(("flat-", "hnsw-")):
                continue
            kind, dim, n, k, r, ns, h = line.split(",")
            key = f"{kind}-d{dim}"
            samples.setdefault(key, []).append(int(ns))
            hashes[arm].setdefault(key, set()).add(h)
        for key, ns in samples.items():
            values[arm].setdefault(key, []).append(statistics.median(ns))
summary = {}
for key in values["baseline"]:
    b = statistics.median(values["baseline"][key])
    c = statistics.median(values["candidate"][key])
    parity = (
        hashes["baseline"][key] == hashes["candidate"][key]
        and len(hashes["baseline"][key]) == 1
    )
    summary[key] = {
        "baseline_ms": b / 1e6,
        "candidate_ms": c / 1e6,
        "less_time_percent": 100 * (1 - c / b),
        "exact_result_parity": parity,
        "baseline_forks_ns": values["baseline"][key],
        "candidate_forks_ns": values["candidate"][key],
    }
(out / "summary.json").write_text(
    json.dumps({"workloads": summary, "searcher_allocations": allocations}, indent=2)
)
print(json.dumps(summary, indent=2))
if not all(v["exact_result_parity"] for v in summary.values()):
    raise SystemExit("RESULT PARITY FAILED")
