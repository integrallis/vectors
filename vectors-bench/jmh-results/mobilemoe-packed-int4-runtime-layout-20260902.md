# MobileMoE packed-INT4 runtime-layout experiment

Date: 2026-09-02

This experiment asks how Java should execute Meta MobileMoE-S QAT expert tensors. The checkpoint
stores each routed-expert projection as signed group-32 INT4 with FP16 scales in an
input-major layout. The product requirement is to consume those bytes in-process without embedding
the upstream runtime or adding a native shim.

## Controlled environment

- Host: `vectors-bench`, AMD EPYC Milan, 4 physical cores / 8 vCPUs, AVX2, 30 GiB
- Runtime: Temurin 25.0.3 HotSpot C2, Vector API at 256 bits
- JMH: two forks, three one-second warmups, five one-second measurements
- Shapes: routed gate/up `768x768` and down `384x768`, batches 1, 4, 8, and 16
- Inputs and weights: deterministic synthetic values using the checkpoint's exact signed INT4,
  group-32 FP16-scale layout

Raw results are retained in
[`mobilemoe-int4-bf16-epyc-20260902.json`](mobilemoe-int4-bf16-epyc-20260902.json) and
[`mobilemoe-q8-epyc-20260902.json`](mobilemoe-q8-epyc-20260902.json).

## Results

Expanding the matrix to BF16 was consistently slower than executing the compact INT4 layout and
would add about 2.1 GiB for all routed experts. It is rejected.

| Batch | Shape | Packed INT4 | Expanded BF16 | BF16 vs INT4 |
| ---: | --- | ---: | ---: | ---: |
| 1 | gate/up | 317.69 us | 513.87 us | 1.62x slower |
| 1 | down | 165.96 us | 260.02 us | 1.57x slower |
| 4 | gate/up | 979.48 us | 2,039.69 us | 2.08x slower |
| 4 | down | 525.42 us | 1,030.89 us | 1.96x slower |
| 8 | gate/up | 1,981.60 us | 4,092.01 us | 2.07x slower |
| 8 | down | 1,043.61 us | 2,056.36 us | 1.97x slower |
| 16 | gate/up | 3,943.57 us | 8,211.21 us | 2.08x slower |
| 16 | down | 2,059.55 us | 4,122.23 us | 2.00x slower |

Transposing and re-encoding each routed projection once into row-major GGUF Q8_0 made the existing
Java Q8-by-Q8 kernel substantially faster. The largest gains appear when the work crosses the
parallel row-dispatch threshold.

| Batch | Shape | Packed INT4 | Prepared Q8_0 | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | gate/up | 317.69 us | 115.21 us | 2.76x |
| 1 | down | 165.96 us | 58.16 us | 2.85x |
| 4 | gate/up | 979.48 us | 327.37 us | 2.99x |
| 4 | down | 525.42 us | 161.97 us | 3.24x |
| 8 | gate/up | 1,981.60 us | 165.86 us | 11.95x |
| 8 | down | 1,043.61 us | 277.26 us | 3.76x |
| 16 | gate/up | 3,943.57 us | 292.84 us | 13.47x |
| 16 | down | 2,059.55 us | 148.18 us | 13.90x |

## Product decision

Models keeps the immutable 681 MiB Safetensors checkpoint mapped and, by default, prepares its
projection weights as Q8_0 in backend-owned native memory. This adds roughly 1.1 GiB for routed
experts rather than the rejected 2.1 GiB BF16 expansion. A
`-Dmodels.mobilemoe.runtimeLayout=packed-int4` compact fallback executes the artifact layout
directly when memory is more important than latency.

For row-major checkpoint matrices, the conversion does not requantize from floating point:
`q8 = q4 * 16` and `q8Scale = originalScale / 16`. That preserves the signed INT4 values while
allowing the common Q8 kernel to consume them. Routed expert tensors require a transpose and a
measured Q8 approximation; the full official PyTorch logit oracle remains above its unchanged
cosine floors.

The full-model progression was 22.58 s TTFT for the first direct implementation, about 9.0 s after
batched prefill and safe INT8 projections, 4.38 s after preparing routed experts, and 2.49 s after
preparing all projections at batch 64. A measured 256-token prefill default cleared the controlled
gate. The final 27-attempt Models report records 27/27 correct answers, 958.0 ms p95 TTFT,
21.83 tokens/second median decode, 2,315.6 ms p95 end-to-end latency, and a
`PRODUCTION_READY` tier on the same host.

This result does not establish that Q8 preparation is best for every QAT format or machine. It is
specific to this layout and the retained Java 25 x86 evidence; the compact implementation remains
the portability and memory fallback.

## JVM implication

Java can run this graph competitively with existing Vector API and FFM memory APIs, but the winning
path must duplicate the weights into an execution-oriented layout. A portable packed signed-INT4
dot product over group scales, together with predictable AArch64 and x86 lowering, could recover
much of the memory advantage without the prepared Q8 copy. This benchmark is a concrete acceptance
case for that JVM capability: match prepared-Q8 throughput while reading the original mapped INT4
layout and preserve the full-model oracle and qualification results.
