# vectors-core

Foundation module for the Vectors library. No internal dependencies.

## Responsibility

- SIMD distance kernels (dot product, L2/Euclidean, cosine similarity) via the Java Vector API (`jdk.incubator.vector`)
- Vector type abstractions (FLOAT32, INT8, BINARY encodings)
- Distance metric interfaces and similarity functions
- Common data structures used across all index modules

## Key Design Decisions

- Float kernels use `SPECIES_PREFERRED`; byte kernels use fixed-width widening tiers
- Full-lane SIMD loops use scalar tails rather than masked tail loads
- Dot product, L2, and cosine kernels use conditional FMA dispatch
- Scalar implementation is available when the Panama Vector implementation is disabled or cannot
  initialize

## Dependencies

- `org.slf4j:slf4j-api` (logging)
