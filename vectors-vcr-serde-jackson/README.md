# vectors-vcr-serde-jackson

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-vcr-serde-jackson/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

VCR cassette serialization using Jackson. Alternative serializer for cassette storage.

## Responsibility

- `JacksonCassetteSerializer` implements `CassetteSerializer` using Jackson's streaming API (JsonGenerator/JsonParser)
- Handles all three `CassetteRecord` types: Embedding, BatchEmbedding, Chat
- Produces the same JSON shape as the Avaje serializer for cross-serializer interoperability

## Key Types

- `JacksonCassetteSerializer` — Jackson streaming-based serializer

## Dependencies

- `vectors-vcr-core` — CassetteSerializer SPI
- Jackson Databind 2.18.2
