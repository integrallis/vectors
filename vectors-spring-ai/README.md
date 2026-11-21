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

One artifact across all supported lines; the application's BOM chooses the framework version at
runtime. The jar is compiled against the 1.1.x baseline, and CI runs those baseline-compiled classes
on 2.1 as well as recompiling per line.

| Spring AI line | Status | Notes |
|----------------|--------|-------|
| 1.0.x – 1.1.x | supported | Baseline compilation target |
| 2.0.x | supported | Spring Framework 7 / Jackson 3 |
| 2.1.x | supported | `VectorStore.upsert(List<EmbeddedDocument>)` and `similaritySearch(float[], SearchRequest)` are supported |

On 2.1 both additions go through Spring AI's own observed template methods: the store implements
the `doUpsert` and `doSimilaritySearch(float[], SearchRequest)` hooks and reports
`supportsUpsert()` / `supportsQueryEmbeddingSearch()`, so they are observed and validated like
`add` and text search. Neither calls the configured `EmbeddingModel`, and a query vector is used as
given: the collection's query prefix is not applied to it. On lines before 2.1 these two operations
do not exist.
