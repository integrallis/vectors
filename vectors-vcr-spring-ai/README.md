# vectors-vcr-spring-ai

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-vcr-spring-ai/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

VCR model wrappers for Spring AI's `EmbeddingModel` and `ChatModel`. Records and replays API calls during test execution.

## Responsibility

- `VCRSpringAIEmbeddingModel` — wraps a Spring AI `EmbeddingModel`, intercepts embed calls for record/replay
- `VCRSpringAIChatModel` — wraps a Spring AI `ChatModel`, intercepts chat calls for record/replay
- `SpringAIModelWrapperProvider` — `ModelWrapperProvider` SPI implementation for automatic `@VCRModel` field discovery
- Call counters generate deterministic cassette keys across test runs
- Mode-aware dispatch: short-circuits on OFF, strict playback throws on missing cassette

## Key Types

- `VCRSpringAIEmbeddingModel` — embedding model wrapper
- `VCRSpringAIChatModel` — chat model wrapper

## Dependencies

- `vectors-vcr-core` — VCR engine
- Spring AI Model 1.1.4 — compile-only
