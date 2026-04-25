# vectors-langchain4j

LangChain4j `EmbeddingStore` adapter backed by a java-vectors `VectorCollection`.

## Responsibility

- `JavaVectorsEmbeddingStore` adapts `VectorCollection` to LangChain4j's `EmbeddingStore<TextSegment>` interface
- Accepts pre-computed `Embedding` objects (no embedding model required)
- Bidirectional metadata conversion between LangChain4j `Metadata` and java-vectors `MetadataValue`
- Filter expression translation from LangChain4j's filter DSL to java-vectors `Filter` AST
- Deletion by ID, by collection, or by filter predicate

## Key Types

- `JavaVectorsEmbeddingStore` — main adapter (builder pattern)
- `MetadataConverter` — lossless type mapping for String, Number, Boolean, List
- `FilterConverter` — translates LangChain4j filter operators to java-vectors `Filter` AST

## Dependencies

- `vectors-db` — VectorCollection API
- LangChain4j Core 1.0.0-beta1 — compile-only
