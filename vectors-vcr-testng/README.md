# vectors-vcr-testng

TestNG listener for VCR record/replay. Mirrors the JUnit 5 extension's functionality using TestNG's listener model.

## Responsibility

- `VCRListener` — implements `ITestListener` and `IInvokedMethodListener` for TestNG lifecycle integration
- `@VCRTestNG` — class-level annotation to enable VCR with mode and data directory
- Per-method mode override via `@VCRRecord` and `@VCRDisabled` annotations
- Automatic `@VCRModel` field wrapping before test method execution
- Auto-registered via `META-INF/services`

## Usage

```java
@VCRTestNG(mode = VCRMode.PLAYBACK_OR_RECORD, dataDir = "src/test/resources/cassettes")
@Listeners(VCRListener.class)
public class MyTest {
    @VCRModel
    EmbeddingModel embeddingModel = new OpenAiEmbeddingModel(...);

    @Test
    public void testEmbedding() {
        // First run: records to cassette. Subsequent runs: replays.
    }
}
```

## Dependencies

- `vectors-vcr-core` — VCR engine
- TestNG 7.10.2 — compile-only
