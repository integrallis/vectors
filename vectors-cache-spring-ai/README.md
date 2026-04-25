# vectors-cache-spring-ai

Caching decorators for Spring AI's `EmbeddingModel` and `ChatModel`. Intercepts calls and serves results from a `VectorCache` using a get-or-embed pattern.

## Responsibility

- `CachingEmbeddingModel` — wraps an `EmbeddingModel`, caches embeddings by normalized text, coalesces batch cache misses into a single delegate call
- `CachingChatModel` — wraps a `ChatModel`, caches `ChatResponse` by prompt and options; streaming calls are forwarded without caching

## Key Types

- `CachingEmbeddingModel` — embedding cache decorator
- `CachingChatModel` — chat response cache decorator

## Dependencies

- `vectors-cache` — VectorCache SPI
- Spring AI Model 1.1.4 — compile-only
