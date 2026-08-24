# vectors-vcr-spring-ai

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-vcr-spring-ai/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

VCR model wrappers for Spring AI's embedding, blocking chat, and streaming chat interfaces. Records and replays API calls during test execution.

## Responsibility

- `VCRSpringAIEmbeddingModel` — wraps a Spring AI `EmbeddingModel`, intercepts embed calls for record/replay
- `VCRSpringAIChatModel` — wraps a Spring AI `ChatModel`, intercepts chat calls for record/replay
- `VCRSpringAIStreamingChatModel` — wraps standalone `StreamingChatModel` implementations
- Complete chat playback: tool calls, generations, token usage, metadata, and ordered stream chunks
- Request-aware replay: model inputs and options are signed; PLAYBACK_OR_RECORD refreshes stale calls
- `SpringAIModelWrapperProvider` — `ModelWrapperProvider` SPI implementation for automatic `@VCRModel` field discovery
- Call counters generate deterministic cassette keys across test runs
- Mode-aware dispatch: short-circuits on OFF, strict playback throws on missing cassette

## Key Types

- `VCRSpringAIEmbeddingModel` — embedding model wrapper
- `VCRSpringAIChatModel` — chat model wrapper
- `VCRSpringAIStreamingChatModel` — standalone streaming chat wrapper

## Dependencies

- `vectors-vcr-core` — VCR engine
- Spring AI Model 1.1.4 — compile-only
