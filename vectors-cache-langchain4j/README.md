# vectors-cache-langchain4j

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-cache-langchain4j/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Caching decorators for LangChain4j's `EmbeddingModel`, `ChatLanguageModel`, and `EmbeddingStore`. Intercepts calls and serves results from a `VectorCache` using a get-or-embed pattern.

## Responsibility

- `CachingEmbeddingModel` — wraps an `EmbeddingModel`, caches embeddings by normalized text, supports both `embed()` and `embedAll()` paths
- `CachingChatModel` — wraps a `ChatLanguageModel`, caches responses for both `generate()` and `doChat()` methods
- `CachingEmbeddingStore` — wraps an `EmbeddingStore`, caches search results by request; write-through semantics invalidate the search cache on mutations

## Key Types

- `CachingEmbeddingModel` — embedding cache decorator
- `CachingChatModel` — chat response cache decorator
- `CachingEmbeddingStore` — embedding store search cache decorator

## Dependencies

- `vectors-cache` — VectorCache SPI
- LangChain4j Core 1.0.0-beta1 — compile-only
