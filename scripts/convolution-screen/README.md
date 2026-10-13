# Convolution screening harness

Four prospectively fixed matched fresh-JVM pairs on each of Linux x86 and ARM, JDK 25.0.3.
This is a diagnostic screen, never final performance or release qualification. The protocol pins
both actual runtime JAR hashes; the candidate source is the runtime-equivalent continuing branch
commit. No regressed experimental implementation is retained in this harness.

The cold counter starts before the first VectorUtil/Panama/helper initialization. Warm timing uses
10,000 warmup calls and 20,000 measured calls, without instrumentation in the numerical proof.
Independent accuracy comparison permits the published 2e-6 absolute output tolerance; history is
bit exact. A cloned cold output and history are repeated after warmup and compared bit for bit.
After timing, both FMA settings also run the actual helper exhaustive numerical proof in separate JVMs.

Every completed pair is uploaded before the next pair. Timeouts retain process logs and receipts.
The analyzer requires both manifests and all 16 arm processes, rejects mixed artifact/workload
provenance and numerical failures, and reports the original metric gates descriptively. Four
pairs cannot establish the full release confidence requirement. No observations are excluded. A predeclared first-pair rejection stops further work if latency,
cold allocation, or warm allocation exceeds its unchanged limit; early stopping cannot pass a gate.

The retired three-arm scalar harness was archived outside the codebases at
`audit-artifacts/convolution-retired-harness-825c912e`, including its SHA-256 inventory.

Before pinning an artifact, run `freeze.py --output /absolute/external/archive` with JAVA_HOME
set to the build JDK. It requires clean runtime sources, checks formatting, performs a clean core
build, and records every source-file hash plus class and disassembled-bytecode hashes. Update
the protocol only from this final receipt. The runner retains the full JAR hash check and also
checks pinned runtime-source and bytecode hashes before executing any measurement.
