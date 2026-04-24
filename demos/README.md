# java-vectors demos

Runnable sample applications demonstrating the library. Each demo is its own Gradle
subproject under `demos/<name>`. Demos are not published as Maven artifacts.

| Module | Path | What it shows |
|---|---|---|
| `:demos:quickstart` | [quickstart/](quickstart/) | Open a collection, add vectors, search — in under 50 lines. |
| `:demos:spring-ai-rag` | [spring-ai-rag/](spring-ai-rag/) | Spring Boot RAG app using `JavaVectorsVectorStore` as a drop-in replacement for `SimpleVectorStore`. |
| `:demos:langchain4j-rag` | [langchain4j-rag/](langchain4j-rag/) | LangChain4j RAG using `JavaVectorsEmbeddingStore` as a drop-in replacement for `InMemoryEmbeddingStore`. |
| `:demos:embedding-cache` | [embedding-cache/](embedding-cache/) | Mmap-persistent get-or-embed cache for LLM workloads; SQ8 quantized; survives restart. |
| `:demos:rerank-after-retrieval` | [rerank-after-retrieval/](rerank-after-retrieval/) | SIMD-accelerated in-VM rescore of candidates retrieved from a remote store. |
| `:demos:vcr-e2e` | [vcr-e2e/](vcr-e2e/) | End-to-end tour of the VCR test harness (JUnit 5 + LangChain4j + Spring AI + manual interceptor). |

## Running

From `vectors/`:

```bash
./gradlew :demos:quickstart:run
./gradlew :demos:spring-ai-rag:bootRun          # requires OPENAI_API_KEY or uses a local embedding model
./gradlew :demos:langchain4j-rag:run
./gradlew :demos:embedding-cache:run
./gradlew :demos:rerank-after-retrieval:run
./gradlew :demos:vcr-e2e:test                    # vcr-e2e is a test-driven demo
```

Every demo applies `--add-modules jdk.incubator.vector` and targets JDK 25.
