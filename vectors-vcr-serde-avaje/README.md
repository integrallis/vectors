# vectors-vcr-serde-avaje

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-vcr-serde-avaje/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

VCR cassette serialization using Avaje Jsonb. Default serializer for cassette storage.

## Responsibility

- `AvajeCassetteSerializer` implements `CassetteSerializer` using Avaje Jsonb's tree API
- Zero-annotation JSON serialization (no annotation processor required)
- Handles all three `CassetteRecord` types: Embedding, BatchEmbedding, Chat
- Produces the same JSON shape as the Jackson serializer for cross-serializer interoperability

## Key Types

- `AvajeCassetteSerializer` — Avaje-based serializer

## Dependencies

- `vectors-vcr-core` — CassetteSerializer SPI
- Avaje Jsonb 3.11
