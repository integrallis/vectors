# vectors-quantization

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-quantization/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Vector quantization implementations for the Vectors library.

## Responsibility

- Scalar quantization (int8, int4)
- Product Quantization (PQ)
- Binary Quantization (BQ)
- Non-uniform Vector Quantization (NVQ)
- RaBitQ
- Quantization rescore pipeline: coarse search with quantized vectors, rescore top candidates with full-precision vectors

## Dependencies

- `vectors-core`
- `vectors-storage`
- `org.slf4j:slf4j-api` (logging)
