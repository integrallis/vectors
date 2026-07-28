# Vectors Jupyter notebooks

Six executable Java notebooks exercise Vectors end to end with the
[DFLib JJava](https://dflib.org/jjava/docs/1.x/) kernel on JDK 25.
The checked-in notebooks include the outputs from their last certified
source-mode execution, so they are useful both as runnable examples and as
rendered, pre-executed documentation.

| Notebook | Topic | Dependency modes |
|---|---|---|
| [01_getting_started.ipynb](01_getting_started.ipynb) | `VectorCollection` basics: FLAT vs HNSW, distance metrics, add, search, and delete | Source, release |
| [02_quantization_tour.ipynb](02_quantization_tour.ipynb) | SQ8, PQ, and BQ compression versus recall | Source, release |
| [03_spring_ai_integration.ipynb](03_spring_ai_integration.ipynb) | `JavaVectorsVectorStore` for Spring AI | Source, release |
| [04_langchain4j_integration.ipynb](04_langchain4j_integration.ipynb) | `JavaVectorsEmbeddingStore` for LangChain4j | Source, release |
| [05_embedding_cache.ipynb](05_embedding_cache.ipynb) | `CachingEmbeddingModel` and `CaffeineVectorCache` | Source, release |
| [06_vcr_test_harness.ipynb](06_vcr_test_harness.ipynb) | Record, replay, and inspect a VCR cassette | Source, release |

## Dependency modes

The notebooks contain no versioned JAR paths. Before Jupyter starts, Gradle
populates `build/notebooks/classpath/`, and the JJava kernel loads that stable
directory through `JJAVA_CLASSPATH`.

- `source` resolves the current Gradle project outputs and their runtime
  dependencies. This is the default for contributors.
- `release` resolves the selected `com.integrallis` artifacts from Maven
  Central. Set `VECTORS_VERSION` once to test a specific release.

You can prepare either classpath without Docker:

```bash
./gradlew prepareNotebookClasspath -PnotebookMode=source
./gradlew prepareNotebookClasspath \
  -PnotebookMode=release \
  -PnotebookVersion=0.1.1
```

Before a release reaches Central, resolve the same artifacts from the local
staging repository:

```bash
./gradlew verifyStagedPublications
./gradlew prepareNotebookClasspath \
  -PnotebookMode=release \
  -PnotebookVersion=0.1.1 \
  -PnotebookRepository=build/staging-deploy
```

## Launch

Docker and Docker Compose are the only host prerequisites.

```bash
cd notebooks
docker compose up --build
```

Jupyter Lab is available at <http://localhost:8888> without a local
development token. To exercise a published release instead:

```bash
VECTORS_NOTEBOOK_MODE=release \
VECTORS_VERSION=0.1.1 \
docker compose up --build
```

## Verify

The test script executes notebooks into `build/notebooks/executed/`, then
inspects every output cell for errors, stderr, suspicious exception text,
unexpected display values, and notebook-specific semantic results. It leaves
the checked-in files unchanged:

```bash
cd notebooks
docker compose build
docker compose run --rm --no-deps jupyter \
  bash /home/jovyan/work/vectors/notebooks/scripts/test-notebooks.sh
```

Release mode executes all six notebooks from the selected Maven artifacts:

```bash
VECTORS_NOTEBOOK_MODE=release \
VECTORS_VERSION=0.1.1 \
docker compose run --rm --no-deps jupyter \
  bash /home/jovyan/work/vectors/notebooks/scripts/test-notebooks.sh
```

`./gradlew verifyNotebooks` supplies the fast static release gate. It rejects
hardcoded module paths, stale dependency versions and APIs, checked-in
execution errors, and a mismatched kernelspec.

After a successful source-mode execution, maintainers update the distributable
notebooks with the certified outputs:

```bash
python3 notebooks/scripts/validate_notebook_outputs.py \
  build/notebooks/executed
python3 notebooks/scripts/update_notebook_outputs.py \
  notebooks build/notebooks/executed
./gradlew verifyNotebooks
```

The update script first proves that cell types, ordering, and source text match.
It then copies only execution counts and outputs, removes volatile execution
timestamps, and records a source SHA-256 that the Gradle gate verifies.

## Runtime

- **Kernel:** DFLib JJava `1.0-a7`, backed by JShell.
- **JDK:** Eclipse Temurin 25.
- **Vector API:** `jdk.incubator.vector` is enabled for the kernel JVM and
  compiler.
- **Native access:** enabled for Arrow's memory implementation.

## Layout

```text
notebooks/
|-- README.md
|-- docker-compose.yml
|-- .env.example
|-- jupyter/
|   |-- Dockerfile
|   |-- kernel.json
|   `-- prepare-classpath.sh
|-- scripts/
|   |-- test-notebooks.sh
|   |-- update_notebook_outputs.py
|   `-- validate_notebook_outputs.py
`-- NN_*.ipynb
```
