# vectors-spring-ai

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-spring-ai/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Spring AI `VectorStore` implementation backed by a java-vectors `VectorCollection`.

## Responsibility

- `JavaVectorsVectorStore` adapts `VectorCollection` to Spring AI's `VectorStore` interface
- Bidirectional metadata conversion between Spring AI `Map<String, Object>` and java-vectors `MetadataValue`
- Filter expression translation from Spring AI's filter DSL to java-vectors `Filter` AST
- Micrometer observation context for search operations

## Key Types

- `JavaVectorsVectorStore` — main adapter (builder pattern)
- `MetadataConverter` — lossless type mapping for String, Number, Boolean, List
- `FilterConverter` — translates EQ, NE, GT, GTE, LT, LTE, IN, NIN, AND, OR, NOT

## Dependencies

- `vectors-db` — VectorCollection API
- Spring AI 1.1.4 (`spring-ai-vector-store`, `spring-ai-model`) — compile-only
- Micrometer Observation 1.14.4 — compile-only
