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
- Spring AI 1.1.x / 2.0.x / 2.1.x (`spring-ai-vector-store`, `spring-ai-model`) — compile-only
- Micrometer Observation 1.14.x / 1.15.x — compile-only

## Spring AI version support

One artifact across all supported lines; the application's BOM chooses the framework version at runtime.

| Spring AI line | Status | Notes |
|----------------|--------|-------|
| 1.0.x – 1.1.x | supported | Baseline compilation target |
| 2.0.x | supported | Spring Framework 7 / Jackson 3 |
| 2.1.x | supported | Adds `upsert()` and `similaritySearch(float[], …)`; both implemented |

## Key types

- `JavaVectorsVectorStore` — main adapter (builder pattern)
  - `add(List<Document>)` — embeds documents via configured `EmbeddingModel`
  - `upsert(List)` — accepts pre-embedded entries reflecting Spring AI 2.1's `EmbeddedDocument` shape
  - `similaritySearch(SearchRequest)` — embeds the query, then searches
  - `similaritySearch(float[], SearchRequest)` — caller-supplied embedding; skips re-embedding (Spring AI 2.1)
- `MetadataConverter` — lossless type mapping for String, Number, Boolean, List
- `FilterConverter` — translates EQ, NE, GT, GTE, LT, LTE, IN, NIN, AND, OR, NOT
