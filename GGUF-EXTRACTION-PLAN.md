# GgufKernel Extraction Plan

## Status: WIP — Requires Tool-Based Refactoring

Manual extraction of ~65 GGUF methods from `VectorUtilSupport` into `GgufKernel` 
failed due to:
1. Method interdependencies (e.g., `ggufQ8_0RowBytes` called by `ggufQ8_0Q8_0MatVecDot`)
2. Overloaded methods with different signatures
3. Private static helpers that must remain accessible
4. Methods that transitively depend on `FloatDistanceKernel` methods

## Recommended Approach

Use IntelliJ IDEA's **Extract Interface** refactoring:
1. Open `VectorUtilSupport.java`
2. Select GGUF methods (lines 178-2667)
3. Refactor → Extract → Interface
4. Name it `GgufKernel`
5. Choose "Declare abstract methods in original interface" (keeps defaults)
6. Update `VectorUtilSupport extends GgufKernel`

This preserves:
- All default implementations
- Private helper visibility
- Cross-method references
- Compile correctness

## After Extraction

The resulting structure should be:
```java
public interface GgufKernel {
    // ~65 method declarations with default scalar implementations
}

public interface VectorUtilSupport extends 
    FloatDistanceKernel, ByteDistanceKernel, SegmentDistanceKernel,
    BinaryDistanceKernel, VectorArithmeticKernel, GgufKernel {
    // Non-GGUF methods (offset arithmetic, GEMV, batchCosine)
}
```

## MFCQI Impact (estimated)

| Metric | Before #114 | After #114 | After GgufKernel |
|--------|-------------|------------|------------------|
| Score  | 0.715       | 0.731      | ~0.745           |
| RFC    | 0.000       | 0.000      | 0.000 (needs composition) |
| Halstead | 0.526     | 0.574      | ~0.600           |

## Note

RFC = 0.000 will persist until `PanamaVectorUtilSupport` is refactored to 
implement individual kernel interfaces directly (composition pattern), rather 
than extending the monolithic `VectorUtilSupport`.
