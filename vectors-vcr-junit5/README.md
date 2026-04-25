# vectors-vcr-junit5

JUnit 5 extension for VCR record/replay. Manages VCR lifecycle via extension callbacks, wraps `@VCRModel` fields, and supports per-test mode overrides.

## Responsibility

- `VCRExtension` — JUnit 5 extension implementing BeforeAll, BeforeTestExecution, TestWatcher, AfterAll
- `@VCRTest` — class-level annotation to enable VCR with mode and data directory
- Per-test mode override via `@VCRRecord` and `@VCRDisabled` annotations
- Automatic `@VCRModel` field wrapping before test execution
- Registry status update on test success/failure
- `CassetteStoreFactory` SPI for custom store creation (loaded via ServiceLoader)

## Usage

```java
@VCRTest(mode = VCRMode.PLAYBACK_OR_RECORD, dataDir = "src/test/resources/cassettes")
@ExtendWith(VCRExtension.class)
class MyTest {
    @VCRModel
    EmbeddingModel embeddingModel = new OpenAiEmbeddingModel(...);

    @Test
    void testEmbedding() {
        // First run: records to cassette. Subsequent runs: replays.
    }
}
```

## Dependencies

- `vectors-vcr-core` — VCR engine
- JUnit Jupiter API 5.11.4 — compile-only
