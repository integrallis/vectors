# Mapped Float16 reranker-weight experiment

Date: 2026-09-06

## Question

Can a Java Vector API kernel execute the IEEE 754 binary16 weights used by
`mixedbread-ai/mxbai-rerank-xsmall-v1` directly from a mapped Safetensors file, avoiding a
one-time expansion from roughly 142 MB of F16 weights to roughly 283 MB of F32 execution weights?

The candidate is retained under `vectors-bench`, not `vectors-core`. It is a reproducer for a
rejected execution layout, not a public API or production dispatch path.

## Environment and protocol

- Host: Intel Core i7-9750H, macOS Darwin 25.6.0, x86_64
- Runtime: Temurin/OpenJDK 25.0.3, HotSpot C2, 256-bit preferred Vector API species
- JMH: 1.37, one fork, three 1-second warmups, five 1-second measurements
- Shapes: the 384-wide attention projections and 1536-wide feed-forward projections from the
  pinned DeBERTa-v2 xsmall checkpoint
- Batches: 1, 32, and 128 pair sequences
- Correctness control: both arms consume exactly the same binary16-rounded weights and F32
  activations; setup rejects an output outside a relative `1e-4` tolerance
- Reproducer:

  ```bash
  ./gradlew :vectors-bench:jmhJar
  java --add-modules jdk.incubator.vector \
    -jar vectors-bench/build/libs/vectors-bench-0.1.20-jmh.jar \
    Float16MatrixBenchmark -rf JSON \
    -rff vectors-bench/build/reports/jmh/float16-matrix-vectorized-local.json
  ```

The raw local result SHA-256 was
`6852b0ce5e1536517e4fb8b77520e9c3094abadacb4c1612a989debad82f890b`.

## Result

Average time in microseconds per complete batch matrix multiply:

| Batch | Shape | Expanded F32 | Mapped F16 | F16 slowdown |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 384x384 | 77.346 | 966.420 | 12.50x |
| 1 | 1536x384 | 336.740 | 7,929.876 | 23.55x |
| 1 | 384x1536 | 221.471 | 10,876.890 | 49.11x |
| 32 | 384x384 | 1,697.788 | 23,121.544 | 13.62x |
| 32 | 1536x384 | 6,065.023 | 91,887.191 | 15.15x |
| 32 | 384x1536 | 4,018.166 | 81,224.505 | 20.21x |
| 128 | 384x384 | 3,949.707 | 83,987.045 | 21.26x |
| 128 | 1536x384 | 24,641.548 | 339,888.377 | 13.79x |
| 128 | 384x1536 | 13,157.565 | 325,787.892 | 24.76x |

A scalar mapped-F16 baseline was already 5.59x to 8.85x slower. The follow-up decoded packed
halves with integer Vector API operations and reused each decoded block across four activations.
That candidate became slower again. Avoiding half the persistent weight bytes does not compensate
for conversion in every projection invocation.

## Decision

Reject mapped F16 as the default execution layout and do not add it to the Vectors API. The first
pure-Java mxbai experiment should map and validate the F16 Safetensors bytes, expand each immutable
matrix once to F32, and then use the established F32 kernels. The expected roughly 283 MB expanded
weight footprint remains acceptable for this 70.8-million-parameter qualification target and must
be measured at the complete-model gate.

This result is also a precise JVM request. Java 25 has scalar binary16 conversion methods, but the
Vector API has no direct binary16 load/convert operation or half-precision matrix primitive. A
future candidate only succeeds if it preserves the same-value control, beats expanded F32 on both
attention and feed-forward shapes, and improves complete reranker throughput without an external
runtime or native inference engine.
