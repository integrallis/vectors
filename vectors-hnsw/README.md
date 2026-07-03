# vectors-hnsw

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-hnsw/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Hierarchical Navigable Small World (HNSW) graph index implemented in Java.

## Responsibility

- HNSW graph construction with configurable M and efConstruction parameters
- Concurrent index building
- Approximate nearest neighbor search with efSearch tuning
- Filtered search support
- Serialization and deserialization of graph structures

## Dependencies

- `vectors-core`
- `vectors-storage`
- `vectors-quantization`
- `org.slf4j:slf4j-api` (logging)
