# 2026-09-16 — Band GEMM (dequantise-to-F32 panel + FMA tile) vs integer kernels

Status: **pre-registration** (written before any A/B run; results sections follow in a later commit).

## Question

Does a pure-Java "dequantise a cache-sized weight panel to F32, then sweep the activations with a
vector FMA register tile" batched matmul (`GgufBatchedMatmulKernel.BAND_F32`, idea from the
Apache-2.0 qxotic JAM `BandGemm`, re-written here) beat the established integer kernels (Q8
activations, int8/int16 lanes, register tiles) for GGUF Q4_K, Q6_K and Q8_0?

## Decision rule (recorded before running)

The band arm is worth productising only if it beats the existing kernel by **>= 10 %** on at least
one LLM shape on some ISA **without losing > 5 %** elsewhere. Otherwise it stays an experiment.

"LLM shape" = Granite 4.1 3B FFN projections (hidden 2560, FFN 8192): up/gate `8192x2560` and down
`2560x8192`, at batch 1 (decode) and batch 512 (prefill). "Elsewhere" = every other measured
format / shape / batch / thread configuration on the same ISA. A result from one ISA does not decide
for another ISA.

Before any productisation, a separate model-level gate is still required (the band arm is not
numerically identical to the integer kernels; see the correctness section).
