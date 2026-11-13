# Cross-Shape Parity for Grouped Quantized Projection (#119)

**Hypothesis:** The existing grouped batched matmul kernels (Q4_K, Q6_K, Q8_0) produce bitwise-identical or tolerance-bounded equal results across batch sizes (1, 2, 4, 8) and whether the projection is executed as grouped (dual/triple) or independent calls. The fused activation→quantized-scratch seam must not introduce shape-dependent drift.

**Decision rule:** For each supported format (Q4_K, Q6_K, Q8_0), grouped projection with batch N must produce the same score (within empirically-derived tolerance) as N independent single-row projections against the same weight panel. Rejection: any shape produces a statistically different result or an off-by-one block boundary error in tail handling.

**Kill criterion:** If we find a format where grouped and independent paths diverge by more than the activation-quantization expected error (approximately 1%), the grouping optimization must be guarded (or the bug fixed) before it becomes default.

## Background

Vectors already supports dual/triple grouped projection (`*GroupedBatchedMatmul`) and the experimental band GEMM (`GgufBatchedMatmulKernel#BAND_F32` and `#BATCH_DISPATCH`). Issue #119 asks us to measure whether the existing kernels preserve arithmetic across shapes before adding new fused seams.

The Strata Qwen3.6 kernels (#119 reference) preserve reduction order while reusing quantized weight loads and fusing activation/quantization boundaries. Before adopting any Strata-inspired optimization, we must verify our existing kernels already satisfy cross-shape parity.

## Protocol

1. **Generate synthetic weights** in each format with a fixed random seed, with panel sizes larger than LLC (so cold-cache behavior dominates, matching the real deployment where weights exceed cache).
2. **Generate activations** with a separate fixed seed, for batch sizes 1, 2, 4, 8, and non-power-of-two tails (3, 5, 7, 9).
3. **Run grouped batched matmul** for each batch size.
4. **Run independent matmul** (batch size 1, N times) for each batch size.
5. **Compare outputs** element-by-element with a tolerance derived from activation-quantization expected error.
6. **Run the same comparison** for both the INTEGER kernel (default) and the BAND_F32 kernel (experimental).
7. **Record** tolerance violations, if any, by format, batch size, and kernel.

## Reproducing

```bash
# Run the parity unit test (no JMH needed; just correctness)
./gradlew :vectors-core:test --tests "com.integrallis.vectors.core.GgufCrossShapeParityTest"
```

## Deliverables

- [ ] `GgufCrossShapeParityTest`: Unit test asserting cross-shape parity for Q4_K, Q6_K, Q8_0
- [ ] JMH microbenchmark comparing grouped vs independent throughput
- [ ] Remote VPS measurements (deferred until hardware is provisioned)
- [ ] Whole-model A/B in Models (coordinated separately; not a Vectors-only deliverable)
