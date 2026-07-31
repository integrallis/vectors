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

## Q4_K four-query register tile gate

The original Q4_K/Q8_K batched kernel reused a weight row across the activation batch but unpacked
the same Q4 nibbles separately for every query. The x86 register-tiled path keeps each unpacked
256-bit weight vector live while multiplying four Q8_K queries. Its four accumulators are explicit
locals because a `FloatVector[]` accumulator array prevents JDK 25 C2 scalar replacement.

On Temurin 25.0.3 and an idle eight-vCPU AMD EPYC-Milan host, the 1024x2048 projection gate used
three forks, three one-second warmups, five one-second measurements, and eight persistent workers:

| Batch | Established kernel | Four-query tile | Change |
| ---: | ---: | ---: | ---: |
| 4 | 0.657 ms/op | 0.488 ms/op | -25.7% |
| 32 | 5.136 ms/op | 3.853 ms/op | -25.0% |

The 99.9% confidence intervals do not overlap. Batch-32 allocation remained noise-equivalent at
466 versus 438 B/op with zero collections. A counterbalanced MiniCPM5 1B Q4_K_M gate then combined
16 trials per revision against the same 688,065,920-byte artifact and 100-word prompt. Median TTFT
fell from 7958.5 to 6780.0 ms (-14.8%), p95 TTFT fell from 8080.0 to 6872.4 ms (-14.9%), and median
prefill rose from 18.83 to 22.10 tok/s (+17.4%). All 32 outputs had the same SHA-256.

Automatic dispatch is limited to x86 with at least 256-bit vectors and batches of four or more.
ARM/SVE retains the established path pending platform evidence. The model-level RSS readings were
not lifecycle-aligned and support no footprint or capacity claim. The machine-readable aggregate
is in `jmh-results/q4k-register-tile-20260731.json`.

## Q8_0 block-major row accumulation gate

The original block-major Q8_0 row kernel updated `out[batch * rows + row]` after every weight
block. At batch 32 those outputs are widely spaced, so a 1024x2048 projection performed 64
scattered read/modify/write cycles per output. The explicit `ROW_ACCUMULATED` kernel instead reuses
one contiguous batch-sized scratch array for the row range and scatters each completed output once.
Arithmetic order and output bits are unchanged.

On GraalVM Java 25 and the controlled eight-vCPU AMD EPYC-Milan host, three warmups and five
measurements reduced the batch-32 block-major JMH path from 20.571 to 4.335 ms/op (-78.9%). The
same local Java 25 HotSpot/C2 comparison was neutral at 17.713 versus 17.480 ms/op, so Vectors keeps
`SCATTERED` as the compatibility default and exposes the optimized strategy explicitly for measured
model/runtime plans. A six-pair SmolLM2 360M Q8_0 gate in Models reduced p50 TTFT by 2.22%, p95
TTFT by 3.67%, and median process CPU by 2.34%; all 30 corresponding outputs were exact.

A separate llama.cpp-style signed-byte pairwise probe lowered one 32-byte dot from 2.566 to 2.291
ns/op on Graal and helped batches 1 through 8, but the repeated batch-32 gate regressed from 20.648
to 20.758 ms/op. That candidate remains unselected pending a decode-specific gate.

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
