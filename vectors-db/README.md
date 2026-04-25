# vectors-db

Embedded vector database facade. Provides a unified API over multiple index backends with optional quantization, metadata filtering, and mmap-persistent storage.

## Responsibility

- `VectorCollection` interface for storing, searching, deleting, and compacting vector documents
- Pluggable index backends: FLAT (brute-force), HNSW, Vamana/DiskANN, IVF_FLAT, IVF_PQ
- Six quantizer options: SQ8, SQ4, PQ, BQ, RaBitQ, NVQ
- Dual-mode operation: in-memory or mmap-persistent with generation-based atomic commits
- Metadata filtering via a composable `Filter` AST (Equals, Range, In, And, Or, Not)
- Tombstone-based deletion with compaction for ordinal reclamation
- Walk-back crash recovery from corrupt or partial generations
- Query result caching (QvCache)

## Key Types

- `VectorCollection` — main API (builder pattern)
- `Document` — record carrying id, vector, optional text, and metadata
- `SearchRequest` / `SearchResult` — query and result types
- `Filter` / `Filters` — metadata filter sealed AST and factory
- `IndexType` — FLAT, HNSW, VAMANA, IVF_FLAT, IVF_PQ
- `QuantizerKind` — NONE, SQ8, SQ4, PQ, BQ, RABITQ, NVQ
- `VectorCollectionConfig` — immutable configuration record

## Dependencies

- `vectors-core` — distance kernels, similarity functions
- `vectors-storage` — off-heap memory, arena management
- `vectors-quantization` — quantizer implementations
- `vectors-hnsw` — HNSW index (implementation scope)
- `vectors-vamana` — Vamana index (implementation scope)
- `vectors-ivf` — IVF indexes (implementation scope)
- Apache Arrow 19.0.0 — batch ingestion/export
