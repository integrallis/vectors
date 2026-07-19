# Mapped K-Quant Long-Offset Gate

Date: 2026-07-18

This report records the acceptance evidence for
`vectors.gguf.mappedKQuantLongOffsets`. Vectors still compiles, tests, and
publishes for Java 25. Java 26 is an optional runtime, not a new minimum.

## Environment

- Branch: `perf/jdk26-mapped-kquant`
- Candidate head: `3cd0233`
- Controlled host: AMD EPYC Milan, 4 physical cores / 8 vCPUs, AVX2, 30 GiB
- Java 25: Temurin 25.0.3+9-LTS
- Java 26: Temurin 26.0.1+8
- Parallel policy: persistent executor, 8 threads, 2 chunks/thread
- JMH: 3 forks, 3 x 1-second warmups, 5 x 1-second measurements
- Storage: read-only file mappings in `Arena.ofShared()`

The benchmark asserts that every mapped segment is accessible from a platform
worker thread. Earlier `Arena.ofConfined()` results were serial and are not
used. Runs observed while an orphaned `llama-cli` consumed CPU were also
discarded.

## Java 26 Kernels

| Kernel and shape | Legacy | Candidate | Change |
| --- | ---: | ---: | ---: |
| Q4_K GEMV, 1024x2048 | 0.165 ms/op | 0.149 ms/op | -9.7% |
| Q5_K GEMV, 1024x2048 | 0.267 ms/op | 0.269 ms/op | +0.7% |
| Q6_K GEMV, 1024x2048 | 0.258 ms/op | 0.253 ms/op | -1.9% |
| Mixed Q4_K/Q4_K/Q6_K, 1536/192/192x1536 | 0.295 ms/op | 0.274 ms/op | -7.1% |
| Q6_K vocabulary projection, 130560x1536 | 24.158 ms/op | 23.680 ms/op | -2.0% |
| Q4_K prefill, batch 32, 4608x1536 | 20.556 ms/op | 17.910 ms/op | -12.9% |

The Q4_K prefill result has non-overlapping 99.9% confidence intervals:
20.556 +/- 0.075 ms/op versus 17.910 +/- 0.066 ms/op.

## Java 25 Kernel

The same mapped Q4_K prefill case improved from 15.787 +/- 0.074 ms/op to
15.519 +/- 0.072 ms/op, or 1.7%. Java 25 remains faster than Java 26 for this
kernel on the controlled host.

## Full Model Gate

MiniCPM5-1B-Q4_K_M used two warmups and ten measured 64-token trials at a
2,048-token context with eight threads. The GGUF SHA-256 is
`81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa`.
Both Java 26 runs reached the benchmark's `OFFLINE` acceptance tier. The
candidate used the production `auto` policy, not a forced experimental mode.

| Java 26.0.1 metric | Legacy | Auto | Change |
| --- | ---: | ---: | ---: |
| p50 decode | 10.2792 tok/s | 10.7403 tok/s | +4.49% |
| p50 TTFT | 11,000.6 ms | 9,825.9 ms | -10.68% |
| p95 TTFT | 11,142.7 ms | 9,962.5 ms | -10.59% |
| p50 prefill | 13.6320 tok/s | 15.2581 tok/s | +11.93% |
| p50 TPOT | 97.198 ms | 93.015 ms | -4.30% |

Linux `perf stat` covered each complete JVM run. Average reported frequency was
3.440 GHz for legacy and 3.428 GHz for auto, while the candidate reduced
task-clock by 8.59%, cycles by 8.93%, retired instructions by 7.06%, cache
misses by 22.23%, and branch misses by 22.70%. All ten corresponding output
hashes matched exactly.

An earlier non-interleaved run reported a 10.77% decode gain. Replaying the
same candidate commit later on the otherwise-idle KVM moved absolute decode
from 12.7896 to 11.56 tok/s. That result is host-frequency/steal drift, not a
defensible kernel effect, and is superseded by the counter-backed gate above.
The earlier Java 25 full-model result (+1.55% decode, -1.02% p50 TTFT) remains
corroborative rather than acceptance evidence; the Java 25 kernel gate is the
non-overlapping JMH result in the preceding section.

Peak `VmHWM` was also unstable across repeated candidate JVMs (850.8-926.2 MB).
At comparable live lifecycle points, `/proc/<pid>/smaps_rollup` measured
812-814 MiB RSS in both modes, split into approximately 216-218 MiB anonymous
and 594 MiB file-backed memory. No memory-capacity improvement or regression is
claimed from these runs; a lifecycle-aligned sampler is required for that gate.

## Retained Policy

The default mode is `auto`. It requires mapped storage and currently enables
only combinations with controlled x86 evidence:

| Format | Java 25 x86 | Java 26+ x86 | Unmeasured architectures |
| --- | --- | --- | --- |
| Q4_K and Q4-led mixed groups | enabled | enabled | legacy |
| Q5_K | legacy | legacy | legacy |
| Q6_K | legacy | enabled | legacy |

Use `-Dvectors.gguf.mappedKQuantLongOffsets=false` for rollback or `true` to
force all implemented long-offset paths during an explicit experiment.
Startup diagnostics print the resolved Q4/Q5/Q6 policy.

## Reproduction

Build the Java 25-compatible benchmark JAR:

```bash
./gradlew :vectors-core:check :vectors-bench:jmhJar
```

Run the mapped Q4_K prefill case with the selected JDK, changing the final
property between `false`, `true`, and `auto`:

```bash
java -jar vectors-bench/build/libs/vectors-bench-0.1.0-SNAPSHOT-jmh.jar \
  com.integrallis.vectors.bench.GgufMappedKQuantMatVecBenchmark.q4KBatched \
  -p rows=4608 -p cols=1536 -p auxiliaryRows=192 -p batchSize=32 \
  -wi 3 -i 5 -w 1s -r 1s -f 3 \
  -jvmArgsAppend '-Dvectors.gguf.parallel=true -Dvectors.gguf.threads=8 -Dvectors.gguf.mappedKQuantLongOffsets=auto'
```
