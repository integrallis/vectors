#!/usr/bin/env python3
"""Run the same benchmark bytecode against two source checkouts, one JVM at a time."""
import argparse
import hashlib
import json
import os
import platform
import tarfile
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--corpus", type=Path, help="Little-endian fbin; graph-only DOT_PRODUCT run")
    inputs.add_argument("--generation", type=Path, help="Pinned collection generation; read-only replay")
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--cadence", type=int, default=25000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--builder", choices=["serial", "concurrent"], default="concurrent")
    args = parser.parse_args()
    if min(args.count, args.threads, args.queries, args.cadence, args.rounds) < 1:
        parser.error("counts, thread count and rounds must be positive")
    if args.builder == "serial" and args.threads != 1:
        parser.error("serial builder requires --threads 1")
    if args.corpus and not args.query_file:
        parser.error("graph runs require --query-file (truth is recomputed for the selected prefix)")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    java = str(Path(os.environ["JAVA_HOME"]) / "bin/java") if "JAVA_HOME" in os.environ else shutil.which("java")
    javac = str(Path(java).parent / "javac")
    provenance = {"arguments": {key: str(value) for key, value in vars(args).items()}}
    provenance["machine"] = {"platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count()}
    provenance["java"] = subprocess.check_output([java, "-version"], stderr=subprocess.STDOUT, text=True)
    classpaths = {}
    with tempfile.TemporaryDirectory(prefix="vectors-perf-") as temporary:
        scratch = Path(temporary)
        init = scratch / "classpath.gradle"
        init.write_text("""allprojects {
  plugins.withId('java') {
    tasks.register('perfClasspath') {
      dependsOn tasks.named('classes'), sourceSets.main.runtimeClasspath
      doLast { println('RUNTIME_CP=' + sourceSets.main.runtimeClasspath.asPath) }
    }
  }
}
""")
        for arm, root in roots.items():
            provenance[arm] = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
            diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
            (output / (arm + ".patch")).write_bytes(diff)
            untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
            provenance[arm]["untracked_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in untracked if name and (root / name).is_file()}
            with tarfile.open(output / (arm + "-untracked.tar.gz"), "w:gz") as archive:
                for name in provenance[arm]["untracked_sha256"]:
                    archive.add(root / name, arcname=name)
            command = [str(root / "gradlew"), "-I", str(init), ":vectors-bench:perfClasspath", "--console=plain"]
            build = subprocess.check_output(command, cwd=root, stderr=subprocess.STDOUT, text=True)
            (output / (arm + "-build.log")).write_text(build)
            original = next(line.removeprefix("RUNTIME_CP=") for line in build.splitlines() if line.startswith("RUNTIME_CP="))
            frozen = output / (arm + "-jars")
            frozen.mkdir()
            entries = []
            for entry in original.split(os.pathsep):
                path = Path(entry)
                if path.is_file() and path.suffix == ".jar":
                    if path.is_relative_to(root):
                        destination = frozen / path.name
                        shutil.copy2(path, destination)
                        entries.append(str(destination))
                    else:
                        entries.append(entry)
            classpaths[arm] = os.pathsep.join(entries)
        classes = output / "harness"
        classes.mkdir()
        names = ["ExactConstructionBenchmark", "DefaultCollectionReplayBenchmark", "CollectionRecallProbe"]
        sources = [roots["candidate"] / "vectors-bench/src/main/java/com/integrallis/vectors/bench" / (name + ".java") for name in names]
        provenance["harness_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        subprocess.run([javac, "--add-modules", "jdk.incubator.vector", "-cp", classpaths["baseline"], "-d", str(classes), *map(str, sources)], check=True)
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        commands = []
        def run(arm, label, name, parameters):
            command = [java, "--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-Xmx6g", "-cp", str(classes) + os.pathsep + classpaths[arm], "com.integrallis.vectors.bench." + name, *map(str, parameters)]
            commands.append(command)
            (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            print(label, flush=True)
            with (output / (label + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        for round_number in range(args.rounds):
            for arm in (["baseline", "candidate"] if round_number % 2 == 0 else ["candidate", "baseline"]):
                label = f"{arm}-{round_number}"
                if args.corpus:
                    run(arm, label, names[0], [args.corpus.resolve(), args.count, args.threads, 42 + round_number, args.builder, args.query_file.resolve(), args.queries])
                else:
                    run(arm, label, names[1], [args.generation.resolve(), output / (label + "-collection"), args.cadence, args.threads, args.count])
        if args.generation:
            for round_number in range(args.rounds):
                for arm in roots:
                    label = f"{arm}-{round_number}"
                    run(arm, label + "-recall", names[2], [args.generation.resolve(), output / (label + "-collection"), args.count, args.queries])


if __name__ == "__main__":
    main()
