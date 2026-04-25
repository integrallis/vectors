# vectors-vcr-semantic-db

VCR cassette storage backed by a java-vectors `VectorCollection`. Uses vector similarity search to find semantically equivalent prompts for non-deterministic response matching.

## Responsibility

- `SemanticCassetteStore` implements `CassetteStore` with dual backing:
  - Exact store for byte-identical cassette retrieval
  - `VectorCollection` (FLAT, cosine) for similarity-based lookup
- Writes fan out to both stores
- Configurable similarity threshold (default: 0.95)
- Embedding extraction from cassette records for vector indexing

## Key Types

- `SemanticCassetteStore` — dual-backed cassette store

## Dependencies

- `vectors-vcr-core` — CassetteStore SPI
- `vectors-db` — VectorCollection for similarity search
