# vectors-hnsw

Hierarchical Navigable Small World (HNSW) graph index. Pure Java replacement for hnswlib/NMSLIB.

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
