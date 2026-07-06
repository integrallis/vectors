<p align="center">
  <img src="media/icons/logo-200.png" alt="vectors logo" width="200">
</p>

# vectors

Embedded vector search and persistence for Java 25.

`vectors` gives a JVM application indexed similarity search, durable local state,
and metadata filtering in-process — no separate vector-database service. It is
built on the JDK Vector API for SIMD distance kernels and ships a full menu of
ANN indexes, quantizers, and a generation-based mmap storage engine, with
first-class Spring AI and LangChain4j adapters.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![JDK 25+](https://img.shields.io/badge/JDK-25%2B-orange.svg)](https://openjdk.org/projects/jdk/25/)
[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

## Where it fits

A JVM application that needs vector search has two common starting points:
in-memory stores like Spring AI's `SimpleVectorStore` or LangChain4j's
`InMemoryEmbeddingStore`, which do not persist or index at scale, and external
vector databases, which add a service to run, secure, and operate.

`vectors` is the layer in between: indexed local search, durable local state,
and metadata filtering that live inside the application process. You get HNSW,
Vamana, IVF-Flat, and IVF-PQ indexes; eight quantizers; atomic generation-based
persistence; and a Spring AI-first collection API — without deploying or
depending on another system.

## Quick start

```kotlin
dependencies {
    implementation("com.integrallis:vectors-db:0.1.0")
}
```

Create, commit, and search a persistent collection:

```java
Path data = Path.of("/var/lib/myapp/vectors");

try (VectorCollection collection = VectorCollection.builder()
    .storagePath(data)
    .dimension(3)
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .build()) {

  collection.add(Document.of("a", new float[] {1.0f, 0.0f, 0.0f}));
  collection.add(Document.of("b", new float[] {0.0f, 1.0f, 0.0f}));
  collection.commit();

  SearchResult result =
      collection.search(SearchRequest.builder(new float[] {0.9f, 0.1f, 0.0f}, 2).build());
}
```

`storagePath` must be absolute. Omit it for an in-memory collection. Adds,
upserts, and deletes are staged until `commit()` unless an auto-commit threshold
is configured.

### Spring AI

```java
VectorCollection collection = VectorCollection.builder()
    .dimension(embeddingModel.dimensions())
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .build();

VectorStore store =
    JavaVectorsVectorStore.builder(embeddingModel, collection).build();
```

### LangChain4j

```java
VectorCollection collection = VectorCollection.builder()
    .dimension(384)
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .build();

EmbeddingStore<TextSegment> store =
    JavaVectorsEmbeddingStore.builder(collection).build();
```

## Capabilities

| Area | Implementation |
|---|---|
| Distance kernels | float, int8, and binary kernels on the JDK Vector API, with scalar fallback |
| Indexes | FLAT, HNSW, Vamana, IVF_FLAT, IVF_PQ |
| Quantization | SQ8, SQ4, FP16, PQ, BQ/BBQ, RaBitQ, NVQ, TurboQuant |
| Persistence | mmap files, atomic generation commits, recovery walk-back, tombstones, compaction |
| Querying | cosine, dot product, Euclidean, MIPS, metadata filters, batch search, hybrid (dense + full-text), MMR diversity re-ranking |
| Integration | Spring AI, LangChain4j, Spring Boot auto-configuration, semantic-cache adapters |

## Runtime requirements

JDK 25 is required — `vectors` is built on the current Vector and Foreign
Function & Memory APIs. The Vector API is incubating in JDK 25, so applications
run with:

```text
--add-modules jdk.incubator.vector
```

The storage engine calls `posix_madvise` through FFM. On classpath deployments,
add the following to keep that optimization and silence restricted-native-access
warnings:

```text
--enable-native-access=ALL-UNNAMED
```

No JNI library is required for the CPU artifacts. The optional Arrow IPC
exporter/ingester additionally needs the opens Arrow's allocator requires:

```text
--add-opens=java.base/java.nio=ALL-UNNAMED
--add-opens=java.base/sun.nio.ch=ALL-UNNAMED
```

## Performance

The SIMD distance kernels run 4.4–6.8× faster than their scalar equivalents on
the committed AVX2 baseline for representative 768- and 1536-dimensional float
kernels. The repository ships the raw JMH output, and a deterministic recall
gate guards against regressions on every build:

- [`vectors-bench/jmh-results/scalar-baseline.txt`](vectors-bench/jmh-results/scalar-baseline.txt)
- [`vectors-bench/src/test/resources/recall-baseline.txt`](vectors-bench/src/test/resources/recall-baseline.txt)

Distance and recall numbers are hardware- and JDK-specific; reproduce them with
`./gradlew :vectors-bench:jmh` and `:vectors-bench:recallGate`.

## Deployment model

`vectors` runs embedded, one in-process writer per persistent collection
directory. That is the design: no service to operate, no network hop on the read
path, and durable state that lives with the application.

- Reach for an external vector database when you need multi-process writers,
  cross-language clients, or managed multi-tenant operations.
- The on-disk format carries explicit version checks. Back up data before
  upgrading across format versions.
- `vectors-gpu` and the cuVS index types require native NVIDIA libraries.

## Modules

Published to Maven Central (Apache-2.0):

- Core: `vectors-core`, `vectors-storage`, `vectors-quantization`
- Indexes & database: `vectors-hnsw`, `vectors-vamana`, `vectors-ivf`, `vectors-db`
- Frameworks: `vectors-spring-ai`, `vectors-langchain4j`, `vectors-spring-boot-starter`
- Caching: `vectors-cache`, `vectors-cache-jcache`, `vectors-cache-langchain4j`,
  `vectors-cache-semantic-db`, `vectors-cache-spring-ai`

Beyond the embedded library, `vectors-distributed`, `vectors-cluster`,
`vectors-server`, and `vectors-gpu` build the object-storage-backed distributed
tier. These are FSL-1.1-ALv2 licensed and convert to Apache-2.0 on their
documented change date; see [`LICENSING.md`](LICENSING.md).

## Building

JDK 25 is required. With SDKMAN:

```bash
sdk use java 25.0.3-tem
./gradlew spotlessCheck build -x :docs:build
./gradlew :vectors-bench:recallGate
./gradlew complianceCheck
```

Additional suites are opt-in because they require Docker or substantial
resources:

```bash
./gradlew integrationTest
./gradlew distributedTest
./gradlew k8sTest
./gradlew chaosTest
./gradlew scaleTest
```

Release mechanics and required secrets are in [`RELEASING.md`](RELEASING.md).

## Documentation

- [`CHANGELOG.md`](CHANGELOG.md)
- [`RELEASING.md`](RELEASING.md)
- [`LICENSING.md`](LICENSING.md)
- [`SECURITY.md`](SECURITY.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

The Maven artifacts are Apache-2.0 licensed. The distributed-tier modules
(`vectors-distributed`, `vectors-cluster`, `vectors-server`, `vectors-gpu`) use
FSL-1.1-ALv2 and convert to Apache-2.0 on their documented change date. See
[`LICENSING.md`](LICENSING.md) for the module-by-module classification.
