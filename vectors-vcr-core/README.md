# vectors-vcr-core

Framework-neutral record/replay engine for AI model API calls. Records embedding and chat responses to cassettes during test runs, then replays them without network access.

## Responsibility

- VCR mode management: PLAYBACK, RECORD, RECORD_NEW, RECORD_FAILED, PLAYBACK_OR_RECORD, OFF
- `CassetteRecord` sealed interface with Embedding, BatchEmbedding, and Chat payload types
- `CassetteStore` — persistence abstraction for cassette storage (exact lookup, CRUD, filtering)
- `VCRRegistry` — tracks per-test recording status (RECORDED, FAILED, MISSING) across runs
- `VCRContext` — runtime state holder (store, registry, mode, call counters)
- Annotations: `@VCRModel` (field wrapping), `@VCRRecord` (force record), `@VCRDisabled` (skip)
- `ModelWrapperProvider` SPI for framework-specific model wrapping (Spring AI, LangChain4j)

## Key Types

- `VCRMode` — operating mode enum
- `CassetteRecord` — sealed payload interface
- `CassetteStore` — cassette persistence SPI
- `CassetteSerializer` — serialization SPI (JSON format)
- `VCRContext` — per-test runtime state
- `VCRRegistry` — cross-run status tracker

## Dependencies

- `vectors-storage` — StorageBackend for cassette and registry persistence
