# Mapped K-Quant Long-Offset Gate

Date: 2026-07-18

This report records the acceptance evidence for
`vectors.gguf.mappedKQuantLongOffsets`. Vectors still compiles, tests, and
publishes for Java 25. Java 26 is an optional runtime, not a new minimum.

## Environment

- Branch: `perf/jdk26-mapped-kquant`
- Candidate head: `a54986f`
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

| Runtime | Metric | Legacy | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| Java 26.0.1 | p50 decode | 11.5456 tok/s | 12.7896 tok/s | +10.77% |
| Java 26.0.1 | p50 TTFT | 10,820.8 ms | 9,427.8 ms | -12.87% |
| Java 26.0.1 | p95 TTFT | 10,998.4 ms | 9,552.0 ms | -13.15% |
| Java 25.0.3 | p50 decode | 13.2951 tok/s | 13.5012 tok/s | +1.55% |
| Java 25.0.3 | p50 TTFT | 8,443.4 ms | 8,356.9 ms | -1.02% |
| Java 25.0.3 | p95 TTFT | 8,557.4 ms | 8,493.4 ms | -0.75% |

All ten corresponding output hashes matched for both runtime comparisons.
Java 26 candidate RSS increased from 841,883,648 to 918,843,392 bytes and
requires separate attribution.

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
