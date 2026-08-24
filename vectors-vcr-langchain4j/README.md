# vectors-vcr-langchain4j

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-vcr-langchain4j/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

VCR model wrappers for LangChain4j's embedding, blocking chat, and streaming chat interfaces. Records and replays API calls during test execution.

## Responsibility

- `VCREmbeddingModel` — wraps a LangChain4j `EmbeddingModel`, intercepts embed/embedAll calls for record/replay
- `VCRChatModel` — wraps a `ChatModel`, intercepts structured chat calls for record/replay
- `VCRStreamingChatModel` — records partial text, thinking, tool calls, and final responses
- Request-aware replay: model inputs, defaults, tools, and generation parameters are signed
- `VCREmbeddingInterceptor` — alternative interceptor-based embedding recording
- `LangChain4jModelWrapperProvider` — `ModelWrapperProvider` SPI implementation for automatic `@VCRModel` field discovery
- Call counters generate deterministic cassette keys across test runs

## Key Types

- `VCREmbeddingModel` — embedding model wrapper
- `VCRChatModel` — chat model wrapper
- `VCRStreamingChatModel` — streaming chat wrapper
- `VCREmbeddingInterceptor` — interceptor variant

## Dependencies

- `vectors-vcr-core` — VCR engine
- LangChain4j Core 1.13.1 — compile-only
