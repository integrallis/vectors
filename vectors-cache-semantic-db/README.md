# vectors-cache-semantic-db

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-cache-semantic-db/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

`SemanticCache` implementation backed by a java-vectors `VectorCollection`. Uses cosine similarity to find near-duplicate prompts for LLM response caching.

## Responsibility

- `VectorDbSemanticCache` implements `SemanticCache` using vector similarity search
- Stores payloads as metadata via a pluggable `PayloadCodec`
- Configurable similarity threshold (default: 0.92 cosine)
- Supports both persistent and in-memory vector collections

## Key Types

- `VectorDbSemanticCache` — main implementation (builder pattern)
- `PayloadCodec` — bidirectional encoder between payload type and String

## Dependencies

- `vectors-cache` — SemanticCache SPI
- `vectors-db` — VectorCollection for storage and search
