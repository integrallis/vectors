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
outputs. A direct dual-only versus dual-plus-triple Q/K/V gate produced only a 0.14% TTFT shift,
three faster and three slower pairs, and 6.68 MB higher median RSS. Models therefore keeps Q8_0
Q/K/V independent. The exact generic triple API remains available for other model-specific gates.

## Q5_0 batched prefill gate

The Q5_0/Q8_0 batch-major kernel closes the last missing batched prefill format used by the tested
mixed-quantized models. On Java 25 and the controlled eight-vCPU AMD EPYC-Milan host, a 1024x2048
matrix with batch 32, three warmups, and five measurements produced:

| Path | Time | Allocation |
| --- | ---: | ---: |
| 32 independent GEMVs | 4.231 ms/op | 1,521.8 B/op |
| Row-local batched kernel | 2.142 ms/op | 471.5 B/op |

The retained kernel is 49.4% faster and allocates 69.0% fewer bytes per operation. Scalar/Panama
and independent-GEMV tests require bit-exact output. A follow-up that decoded each packed weight
block into four `IntVector` values before traversing the activation batch regressed to 2.955 ms/op,
37.9% slower than the retained helper-based loop, with effectively unchanged allocation. That form
was removed; keeping the decoded vectors live across the batch loop creates a worse compiled shape
on this JDK/CPU.

## Dependencies

- All Vectors library modules
- `org.openjdk.jmh:jmh-core` (benchmarking framework)
