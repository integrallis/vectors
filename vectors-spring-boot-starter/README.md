# vectors-spring-boot-starter

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-spring-boot-starter/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Spring Boot auto-configuration for java-vectors. Registers `VectorCollection` and optional `JavaVectorsVectorStore` beans from application properties.

## Responsibility

- Binds `java-vectors.*` YAML/properties to `VectorCollectionBuilder`
- Auto-configures a `VectorCollection` singleton bean
- Conditionally wires `JavaVectorsVectorStore` when an `EmbeddingModel` is on the classpath
- All beans are `@ConditionalOnMissingBean` for easy overrides

## Quick start

With an `EmbeddingModel` bean on the classpath, adding the dependency is enough — `metric` defaults
to `COSINE` and `dimension` is inferred from `EmbeddingModel.dimensions()`, so no `java-vectors.*`
configuration is required:

```java
@Autowired
VectorStore vectorStore; // auto-configured; no VectorCollection bean to declare
```

## Configuration

Every setting is optional; override defaults as needed:

```yaml
java-vectors:
  dimension: 1536          # optional — inferred from EmbeddingModel.dimensions() when unset
  metric: COSINE           # optional — defaults to COSINE
  index-type: HNSW         # defaults to FLAT
  quantizer: SQ8           # defaults to NONE
  storage-path: /var/lib/vectors/my-collection
  commit-after-add: true   # commit after each store.add(); set false for batch ingestion
  hnsw:
    m: 16
    ef-construction: 200
  vamana:
    max-degree: 64
    search-list-size: 128
    alpha: 1.2
```

Without an `EmbeddingModel` on the classpath, `java-vectors.dimension` is required; the context
fails at startup with a clear message if it is missing.

## Key Types

- `JavaVectorsAutoConfiguration` — auto-configuration with nested Spring AI configuration
- `JavaVectorsProperties` — property binding class with nested HNSW, Vamana, IVF, PQ property groups

## Dependencies

- `vectors-spring-ai` — VectorStore adapter
- Spring Boot Autoconfigure 3.4.5 — compile-only
- Spring AI 1.1.4 — compile-only
