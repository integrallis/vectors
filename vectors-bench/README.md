# vectors-bench

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-bench/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Benchmarks for the Vectors library.

## Responsibility

- JMH microbenchmarks for SIMD distance kernels
- ANN-Benchmarks-aligned macrobenchmark harness measuring recall@k, QPS, and latency
- Comparative benchmarks across index types and quantization strategies

## Running

```bash
./gradlew :vectors-bench:jmh
```

## Grouped GGUF projection gates

Grouped batched GGUF APIs are retained by quantization format only after an exact real-model gate
in Models. On Java 25 and an eight-vCPU AMD EPYC-Milan host, the Q4_0 dual/triple path improved
Qwen3 0.6B median TTFT from 3127.92 to 3096.45 ms across 18 trials per mode, with identical token
counts and output hashes in every counterbalanced pair. A ten-prompt allocation JFR with five
warmups measured a 9.99 MB total allocation difference for the complete process, ruling out a
steady per-prefill allocation penalty. The previously retained mixed Q4_K/Q4_K/Q6_K path improved
MiniCPM5 1B median TTFT from 8330.16 to 7948.31 ms.

These results do not imply that every projection shape benefits from grouped dispatch. A subsequent
SmolLM2 360M gate retained Q8_0 batched dual dispatch for gate/up: median TTFT improved from 2612.09
to 2581.73 ms and prefill from 60.22 to 61.01 tok/s across 18 trials per mode, with exact paired
outputs. Q8_0 triple Q/K/V dispatch remains disabled in Models until its independent gate passes.

## Dependencies

- All Vectors library modules
- `org.openjdk.jmh:jmh-core` (benchmarking framework)
