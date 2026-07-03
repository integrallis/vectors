<p align="center">
  <img src="media/icons/logo-200.png" alt="vectors logo" width="200">
</p>

# vectors

Embedded vector search and persistence for Java 25.

`vectors` provides Vector API distance kernels, ANN indexes, quantization,
generation-based mmap persistence, metadata filtering, and adapters for Spring
AI and LangChain4j. It runs inside the application process and does not require
a separate vector-database service.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![JDK 25+](https://img.shields.io/badge/JDK-25%2B-orange.svg)](https://openjdk.org/projects/jdk/25/)

> **Status: 0.1.0 release candidate, not yet released.** The supported release
> target is the single-process CPU library and its framework/cache adapters.
> Distributed, server, Studio, optimizer, and GPU modules are experimental and
> are not published in 0.1.x.

## Where it fits

Spring AI documents `SimpleVectorStore` as an in-memory implementation intended
for testing rather than production. LangChain4j's `InMemoryEmbeddingStore` is a
simple in-process store with JSON serialization. Those are useful defaults.
External services remain the right answer for managed scale, high availability,
multi-process writers, and cross-language access.

`vectors` targets the space between those choices: a Java application that
wants indexed local search, durable local state, and metadata filtering without
introducing another service.

JVector and Lucene are established, capable Java alternatives. In particular,
JVector already provides a mature Vamana implementation, disk-backed graphs,
quantization, and a LangChain4j integration. The narrower differentiation here
is a Spring AI-first embedded collection API, generation-based persistence,
metadata filters, and a choice of HNSW, Vamana, IVF-Flat, and IVF-PQ indexes.

## 0.1.x scope

The Maven Central publication allowlist contains:

- Core: `vectors-core`, `vectors-storage`, `vectors-quantization`
- Indexes/database: `vectors-hnsw`, `vectors-vamana`, `vectors-ivf`,
  `vectors-db`
- Frameworks: `vectors-spring-ai`, `vectors-langchain4j`,
  `vectors-spring-boot-starter`
- Caching: `vectors-cache`, `vectors-cache-jcache`,
  `vectors-cache-langchain4j`, `vectors-cache-semantic-db`,
  `vectors-cache-spring-ai`

Anything not listed above is excluded from the 0.1.x Maven publication even if
its source is present in this repository.

## Quick start

The coordinates below become available after 0.1.0 is published to Maven
Central:

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

## Implemented capabilities

| Area | 0.1.x implementation |
|---|---|
| Distance kernels | float, int8, and binary kernels using the incubating JDK Vector API with scalar fallback |
| Indexes | FLAT, HNSW, Vamana, IVF_FLAT, IVF_PQ |
| Quantization | SQ8, SQ4, FP16, PQ, BQ/BBQ, RaBitQ, NVQ, TurboQuant |
| Persistence | mmap files, atomic generation commits, recovery walk-back, tombstones, compaction |
| Querying | cosine, dot product, Euclidean distance, MIPS, metadata filters, batch search |
| Integration | Spring AI, LangChain4j, Spring Boot auto-configuration, semantic-cache adapters |

The Vector API remains incubating in JDK 25, so applications must run with:

```text
--add-modules jdk.incubator.vector
```

The storage module opportunistically calls `posix_madvise` through FFM. On
classpath deployments, use the following to avoid restricted-native-access
warnings and retain that optimization:

```text
--enable-native-access=ALL-UNNAMED
```

No JNI library is required for the 0.1.x CPU artifacts.

Applications that use the optional Arrow IPC exporter/ingester also need the
opens required by Arrow's unsafe allocator:

```text
--add-opens=java.base/java.nio=ALL-UNNAMED
--add-opens=java.base/sun.nio.ch=ALL-UNNAMED
```

## Performance evidence

The repository contains JMH output rather than a marketing-only benchmark
table. The committed AVX2 baseline shows roughly 4.4–6.8× SIMD/scalar throughput
for representative 768- and 1536-dimensional float kernels. Results are
hardware- and JDK-specific:

- [`vectors-bench/jmh-results/scalar-baseline.txt`](vectors-bench/jmh-results/scalar-baseline.txt)
- [`vectors-bench/src/test/resources/recall-baseline.txt`](vectors-bench/src/test/resources/recall-baseline.txt)

The recall file is a deterministic regression gate. It tracks drift on a
synthetic corpus; it is not an ANN-Benchmarks leaderboard result. The project
does not currently claim a general vector-count ceiling or a universal query
latency.

## Operational boundaries

- One persistent collection directory has one in-process writer. There is no
  cross-process file lock; coordinating multiple processes requires an external
  lock and is not a supported 0.1.x deployment.
- CPU libraries are the release target. `vectors-gpu` and the cuVS index types
  require native NVIDIA libraries and are experimental.
- `vectors-distributed`, `vectors-cluster`, and `vectors-server` are
  experimental source modules, not a production HA product.
- The on-disk format has explicit version checks, but 0.1.x does not promise
  perpetual forward compatibility. Back up data before upgrading.
- JDK 25 is required. This deliberately trades broad JVM compatibility for the
  current Vector and FFM APIs.

Use an external vector database when you need multi-process writes, managed
backups, production HA, tenant isolation, or non-JVM clients.

## Module status

| Module group | Status |
|---|---|
| Core, storage, quantization | 0.1.x candidate |
| HNSW, Vamana, IVF, embedded DB | 0.1.x candidate |
| Spring AI, LangChain4j, Spring Boot | 0.1.x candidate |
| Cache modules | 0.1.x candidate |
| VCR, hybrid, ingest, router, text sidecars | implemented but excluded from 0.1.x publication |
| Distributed, cluster, server, GPU | experimental; FSL-licensed where documented |
| Studio and optimizer | development tooling; not published |

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

Release mechanics and required secrets are documented in
[`RELEASING.md`](RELEASING.md).

## Project documentation

- [`CHANGELOG.md`](CHANGELOG.md)
- [`RELEASING.md`](RELEASING.md)
- [`COMMUNITY-LAUNCH.md`](COMMUNITY-LAUNCH.md)
- [`LICENSING.md`](LICENSING.md)
- [`SECURITY.md`](SECURITY.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`vectors-distributed-design.md`](vectors-distributed-design.md) — design
  draft, not a statement of released capability

The old competitive analysis is retained only as historical research and must
not be used as a current product comparison until it is rewritten against
released versions and reproducible evidence.

## License

The 0.1.x Maven artifacts are Apache-2.0 licensed. Experimental
`vectors-distributed`, `vectors-cluster`, `vectors-server`, and `vectors-gpu`
use FSL-1.1-ALv2 and convert to Apache-2.0 on their documented change date.
See [`LICENSING.md`](LICENSING.md) for the module-by-module classification.
