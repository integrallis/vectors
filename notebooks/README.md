# java-vectors Jupyter notebooks

Six Jupyter notebooks that walk through the library end-to-end using the
[DFLib JJava](https://dflib.org/jjava/docs/1.x/) kernel (Java in Jupyter, JDK 25).

| Notebook | Topic |
|---|---|
| [01_getting_started.ipynb](01_getting_started.ipynb) | `VectorCollection` basics: FLAT vs HNSW, distance metrics, add / search / delete. |
| [02_quantization_tour.ipynb](02_quantization_tour.ipynb) | SQ8 → PQ → BQ compression-vs-recall on a small synthetic dataset. |
| [03_spring_ai_integration.ipynb](03_spring_ai_integration.ipynb) | `JavaVectorsVectorStore` for Spring AI: add documents, similarity search. |
| [04_langchain4j_integration.ipynb](04_langchain4j_integration.ipynb) | `JavaVectorsEmbeddingStore` for LangChain4j: add `TextSegment`s, search, remove. |
| [05_embedding_cache.ipynb](05_embedding_cache.ipynb) | `CachingEmbeddingModel` + `CaffeineVectorCache` hit-rate walkthrough. |
| [06_vcr_test_harness.ipynb](06_vcr_test_harness.ipynb) | VCR: record a cassette, replay it, inspect the on-disk layout. |

## Prerequisites

- Docker + Docker Compose (tested with Docker 27+).
- A local `./gradlew build -x :docs:build` completed at least once — the
  notebooks pick up the compiled jars from each module's `build/libs/`
  directory.

## Launch

```bash
# From the repository root:
cd vectors
./gradlew build -x :docs:build -x test       # builds jars the notebooks will consume
cd notebooks
cp .env.example .env                           # optional: populate OPENAI_API_KEY, etc.
docker compose up --build
```

Jupyter Lab will be available at <http://localhost:8888>. The access token is
printed to the console on startup. Open any notebook under `/home/jovyan/work/`.

## Implementation notes

- **Kernel:** DFLib JJava `1.0-a7` (January 2026). JShell-backed Java REPL.
- **JDK:** Eclipse Temurin 25.
- **Classpath:** Each notebook pulls jars from `/home/jovyan/work/vectors/vectors-*/build/libs/*.jar`
  via the `%jars` magic. Re-run `./gradlew build` on the host whenever you
  change library source.
- **Vector API:** `--add-modules jdk.incubator.vector` is added to the kernel
  JVM arguments in `jupyter/kernel.json`.
- **`%maven` magic:** Still available for third-party dependencies
  (Spring AI, LangChain4j, Caffeine). Internal modules use `%jars`.

## Layout

```
notebooks/
├── README.md
├── docker-compose.yml
├── .env.example
├── jupyter/
│   ├── Dockerfile
│   ├── kernel.json      # JJava kernelspec (adds --enable-native-access + Vector API)
│   └── install.sh       # Called from the Dockerfile; installs JJava
└── NN_*.ipynb           # six notebooks
```
