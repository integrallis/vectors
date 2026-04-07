# vectors-core

Foundation module for the Vectors library. No internal dependencies.

## Responsibility

- SIMD distance kernels (dot product, L2/Euclidean, cosine similarity) via the Java Vector API (`jdk.incubator.vector`)
- Vector type abstractions (FLOAT32, INT8, BINARY encodings)
- Distance metric interfaces and similarity functions
- Common data structures used across all index modules

## Key Design Decisions

- Uses `SPECIES_PREFERRED` for SIMD portability across hardware
- Full-lane loop with scalar tail (no masked operations)
- `fma()` for fused multiply-add in dot product kernels
- Scalar fallback for platforms without SIMD support

## Dependencies

- `org.slf4j:slf4j-api` (logging)
