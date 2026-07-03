# vectors-spring-boot-starter

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-spring-boot-starter/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Spring Boot auto-configuration for java-vectors. Registers `VectorCollection` and optional `JavaVectorsVectorStore` beans from application properties.

## Responsibility

- Binds `java-vectors.*` YAML/properties to `VectorCollectionBuilder`
- Auto-configures a `VectorCollection` singleton bean
- Conditionally wires `JavaVectorsVectorStore` when an `EmbeddingModel` is on the classpath
- All beans are `@ConditionalOnMissingBean` for easy overrides

## Configuration

```yaml
java-vectors:
  dimension: 1536
  metric: COSINE
  index-type: HNSW
  quantizer: SQ8
  storage-path: /var/lib/vectors/my-collection
  hnsw:
    m: 16
    ef-construction: 200
  vamana:
    max-degree: 64
    search-list-size: 128
    alpha: 1.2
```

## Key Types

- `JavaVectorsAutoConfiguration` — auto-configuration with nested Spring AI configuration
- `JavaVectorsProperties` — property binding class with nested HNSW, Vamana, IVF, PQ property groups

## Dependencies

- `vectors-spring-ai` — VectorStore adapter
- Spring Boot Autoconfigure 3.4.5 — compile-only
- Spring AI 1.1.4 — compile-only
