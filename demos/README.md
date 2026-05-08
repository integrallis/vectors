# java-vectors demos

Runnable sample applications demonstrating the library. Each demo is its own Gradle
subproject under `demos/<name>`. Demos are not published as Maven artifacts.

| Module | Path | What it shows |
|---|---|---|
| `:demos:quickstart` | [quickstart/](quickstart/) | Open a collection, add vectors, search — in under 50 lines. |
| `:demos:spring-ai-rag` | [spring-ai-rag/](spring-ai-rag/) | Spring Boot RAG app using `JavaVectorsVectorStore` as a drop-in replacement for `SimpleVectorStore`. |
| `:demos:langchain4j-rag` | [langchain4j-rag/](langchain4j-rag/) | LangChain4j RAG using `JavaVectorsEmbeddingStore` as a drop-in replacement for `InMemoryEmbeddingStore`. |
| `:demos:embedding-cache` | [embedding-cache/](embedding-cache/) | Decorator-based get-or-embed cache for LLM pipelines — `CachingEmbeddingModel` (vectors-cache-langchain4j) + Caffeine. |
| `:demos:semantic-cache` | [semantic-cache/](semantic-cache/) | Similarity-threshold cache for near-duplicate questions — `VectorDbSemanticCache` over a FLAT/COSINE collection. |
| `:demos:rerank-after-retrieval` | [rerank-after-retrieval/](rerank-after-retrieval/) | SIMD-accelerated in-VM rescore of candidates retrieved from a remote store. |
| `:demos:server-client` | [server-client/](server-client/) | End-to-end HTTP round-trip against an embedded vectors-server on an ephemeral port. |
| `:demos:vcr-e2e` | [vcr-e2e/](vcr-e2e/) | End-to-end tour of the VCR test harness (JUnit 5 + LangChain4j + Spring AI + manual interceptor). |
| `:demos:studio-r2-sidecart` | [studio-r2-sidecart/](studio-r2-sidecart/) | CLI seeder built on `vectors-ingest`'s `BulkIngestor` — embeds a 24-doc corpus into Cloudflare R2 (vectors) plus an H2 or D1 sidecart (text). |

## Running

From `vectors/`:

```bash
./gradlew :demos:quickstart:run
./gradlew :demos:spring-ai-rag:run               # requires OPENAI_API_KEY or uses a local embedding model
./gradlew :demos:langchain4j-rag:run
./gradlew :demos:embedding-cache:run
./gradlew :demos:semantic-cache:run
./gradlew :demos:rerank-after-retrieval:run
./gradlew :demos:server-client:run
./gradlew :demos:vcr-e2e:test                    # vcr-e2e is a test-driven demo
./gradlew :demos:studio-r2-sidecart:runSeed --args="--help"   # see the full flag list
```

Every demo applies `--add-modules jdk.incubator.vector` and targets JDK 25.
