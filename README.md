# java-vectors

```
      ██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ███████╗     ██╗
      ██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝     ╚██╗
█████╗██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝███████╗█████╗╚██╗
╚════╝╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗╚════██║╚════╝██╔╝
       ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████║     ██╔╝
        ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝     ╚═╝
```

> **The Java-native vector database that scales from a single JAR to a distributed cluster — and beyond to GPU — without ever leaving the JVM.**
>
> Pure Java. Zero JNI. JDK 25+.
> SIMD-native kernels, HNSW + Vamana indexing, 8 quantizers (SIGMOD 2024/2025 state of the art), portable mmap persistence, drop-in adapters for **Spring AI** and **LangChain4j**, and a defined path to embedded clustering and cuVS-powered GPU search.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: FSL-1.1-ALv2](https://img.shields.io/badge/License-FSL--1.1--ALv2-purple.svg)](LICENSE-FSL)
[![JDK 25+](https://img.shields.io/badge/JDK-25%2B-orange.svg)](https://openjdk.org/projects/jdk/25/)
[![Pure Java](https://img.shields.io/badge/Pure%20Java-no%20JNI-brightgreen.svg)]()

---

## The pitch in 60 seconds

**Spring AI** and **LangChain4j** both ship a trivial vector store (`SimpleVectorStore`, `InMemoryEmbeddingStore`). Both are explicitly **"not for production"**: O(n) scalar search, JSON persistence, cosine-only, practical ceiling ~50K vectors. The next step today is provisioning a separate database (Pinecone, Qdrant, Redis, pgvector) — infrastructure, a network hop, and $600–1,500/year of hosting.

**java-vectors is the missing rung on the ladder** — the Chroma / SQLite of the Java AI stack:

- **Embed it** like `SimpleVectorStore`: one JAR set, no database process, `<1 ms` in-process queries.
- **Scale it** with HNSW / Vamana indexing, 8 quantizers, mmap persistence, metadata filters — to 10M+ vectors on a single JVM.
- **Cluster it** when you outgrow one node: `vectors-distributed` follows the EclipseStore + Eclipse Data Grid pattern (embedded cluster members auto-join via gossip, no separate DB process).
- **Accelerate it** on GPU via `vectors-gpu`: Panama-FFM bindings to NVIDIA cuVS (CAGRA, IVF-PQ). Same pattern Apache Lucene and Elasticsearch adopted in 2025.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Your Spring AI / LangChain4j / plain Java application               │
│                                                                      │
│    VectorStore / EmbeddingStore  ──(unchanged interface)──►          │
│                                                                      │
│           ┌────────────────────────────────────────────┐             │
│           │            java-vectors                    │             │
│           │                                            │             │
│           │  SINGLE JAR           CLUSTER          GPU │             │
│           │  ───────────          ───────          ─── │             │
│           │  vectors-db     →  vectors-          →  vectors-         │
│           │  mmap, HNSW,       distributed         gpu               │
│           │  SIMD, PQ, filters  gossip+Raft        cuVS via Panama   │
│           │                     segment shards                       │
│           │                                            │             │
│           │  same API across all three tiers           │             │
│           └────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

**Same language. Same runtime. Same API. Same JAR. Different deployment.**

## Why it exists

### The gap in the Java ecosystem

|  | `SimpleVectorStore` (Spring AI) | `InMemoryEmbeddingStore` (LangChain4j) | **java-vectors** |
|--|--|--|--|
| Search | O(n) brute-force | O(n) brute-force | **HNSW / Vamana / Fused ADC** |
| SIMD kernels | none (scalar) | none (scalar) | **Vector API, 3–16× scalar** |
| Indexing | none | none | **HNSW + Vamana** |
| Quantization | none | none | **8 quantizers (SQ, PQ, BQ, RaBitQ, Ext-RaBitQ, NVQ, Turbo)** |
| Persistence | JSON file | JSON file | **mmap + generation commits + walk-back recovery** |
| Distance metrics | cosine only | cosine only | **L2 / cosine / dot / MIPS** |
| Metadata filtering | basic | basic | **structured filter expressions** |
| Practical ceiling | ~10–50K vectors | ~10–50K vectors | **10M+ (HNSW), 100M+ with PQ** |
| Query latency @ 100K/1536d | 20–170 ms | 20–170 ms | **<1 ms** |
| Memory @ 100K/1536d | ~586 MB on-heap | ~586 MB on-heap | **~147 MB off-heap (SQ8)** |

### The operational value

| Option | Infra cost (monthly) | Query latency | Ops complexity | Practical ceiling |
|---|---|---|---|---|
| Pinecone Starter | $70+ | 15–30 ms (network) | zero | unlimited (managed) |
| Self-hosted Qdrant on EC2 | $30–80 + labour | 5–15 ms (LAN) | medium | multi-million |
| pgvector on existing RDS | ~$0 marginal | 10–50 ms | low | ~5M |
| `SimpleVectorStore` | $0 | 20–170 ms (O(n)) | zero | ~50K |
| **java-vectors embedded** | **$0** | **<1 ms** | **zero** | **10M+** |

## Quick start

### Spring AI — drop-in replacement for `SimpleVectorStore`

```java
VectorCollection collection = VectorCollection.builder()
    .dimension(embeddingModel.dimensions())
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .quantizer(QuantizerKind.SQ8)
    .build();

VectorStore store = JavaVectorsVectorStore.builder(embeddingModel, collection).build();
```

Same `VectorStore` interface, same Spring AI application code — now with HNSW indexing, SIMD-accelerated search, scalar quantization, and mmap persistence.

### LangChain4j — drop-in replacement for `InMemoryEmbeddingStore`

```java
VectorCollection collection = VectorCollection.builder()
    .dimension(384)
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .build();

EmbeddingStore<TextSegment> store = JavaVectorsEmbeddingStore.builder(collection).build();
```

### Use only the layer you need

| Goal | Artefact |
|---|---|
| SIMD distance kernels in any JVM app | `vectors-core` |
| Client-side quantization before a remote DB | `vectors-core` + `vectors-quantization` |
| Custom ANN index | `vectors-core` + `vectors-quantization` + `vectors-hnsw` *or* `vectors-vamana` |
| Embedded vector database | `vectors-db` (pulls everything in) |
| Spring AI integration | `vectors-spring-ai` |
| LangChain4j integration | `vectors-langchain4j` |

### Client-side quantization for remote vector DBs

Even if you keep Qdrant / Redis / Milvus / Elasticsearch / pgvector as your backend, `vectors-quantization` trims bandwidth and storage:

```java
ScalarQuantizer sq = ScalarQuantizer.train(dataset, ScalarBits.INT8);
byte[] quantized = sq.encode(floatVector);   // -> 75% bandwidth reduction
```

| Quantizer | Compression | Recall impact | Bandwidth saved |
|---|---|---|---|
| Scalar int8 | 4× | ~1.5 pp drop | 75% |
| Product Quantization | 10–90× | 2–5 pp drop | 90%+ |
| Binary (sign-bit) | 32× | 5–10 pp (no rescore) | 97% |
| RaBitQ (1-bit + corrections) | 32× | <2 pp (with rescore) | ~90% |

## Persistence that travels with your app

`vectors-db` uses **mmap-based storage with generation-based atomic commits** and walk-back crash recovery. One directory is one database. It moves between hosts by `cp -r`. This opens use cases that an external database cannot serve as naturally.

### 1. Embedding cache for LLM workloads

Every production LLM pipeline recomputes embeddings. A mmap-persistent `VectorCollection` beside the app is a zero-infra embedding cache: `get-or-embed` on miss, sub-millisecond recall on hit, survives restarts, and scales to tens of millions of cached embeddings on a commodity box.

```java
VectorCollection cache = VectorCollection.builder()
    .dataDir(Path.of("/var/lib/myapp/embedding-cache"))
    .dimension(1536)
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .quantizer(QuantizerKind.SQ8)    // 4× smaller on disk, identical recall after rescore
    .build();
```

### 2. In-VM reranking after a network retrieval

Remote vector databases (Pinecone, Qdrant, managed Elasticsearch) return ~100–1000 candidates. Reranking with a higher-fidelity metric, a custom cross-encoder score, or a learned linear head is latency-critical and needs **tight loops with float32 precision**. `vectors-core` and `vectors-quantization` give you SIMD-accelerated distance and rescore kernels without allocating a database.

```java
// From a Qdrant/Pinecone response of N candidates and a fresh query:
float[] scores = VectorUtil.dotProductBatch(query, candidateVectors);   // SIMD, 3-16x scalar
int[] topK = TopK.select(scores, k);                                    // quickselect
```

### 3. Portable, single-directory format

The on-disk layout is documented and stable; collections copy, tar, and ship across hosts. No server, no cluster, no network port. This is the `SQLite` model applied to vectors.

### 4. Distributable by construction

The same `vectors-db` format is the on-disk unit of sharding for `vectors-distributed`: a cluster node owns a set of segments, each segment is a mmap directory, and nodes exchange segments during rebalance. Start single-node; scale out without rewriting data.

## Measured performance

All numbers from the ANN-Benchmarks harness (`:vectors-bench:recallQps`) on **Apple Silicon M3, NEON 128-bit, JDK 25, single thread**. Full CSV: `research/data/results/recall-qps.csv`.

| Dataset | N · Dim | Index | Recall@10 | QPS | P50 |
|---|---|---|---|---|---|
| fashion-mnist-784-euclidean | 60K · 784 | HNSW M=16 ef=128 | **0.9996** | 3,514 | 277 µs |
| fashion-mnist-784-euclidean | 60K · 784 | HNSW M=16 ef=16 (speed) | 0.988 | **13,758** | 71 µs |
| fashion-mnist-784-euclidean | 60K · 784 | Vamana L=128 R=32 α=1.0 | 0.999 | 6,185 | 159 µs |
| fashion-mnist-784-euclidean | 60K · 784 | Fused ADC pq=49 (**16× compressed**) | 0.997 | 4,352 | 221 µs |
| fashion-mnist-784-euclidean | 60K · 784 | Fused ADC pq=16 (**49× compressed**) | 0.991 | 4,668 | 207 µs |
| sift-128-euclidean | 100K · 128 | HNSW M=16 ef=128 | 0.993 | 3,017 | 331 µs |
| sift-128-euclidean | 100K · 128 | HNSW M=32 ef=64 | 0.990 | 4,370 | 236 µs |
| glove-100-angular | 1.18M · 100 | HNSW M=16 ef=64 | 0.774 | 3,841 | 246 µs |

**Concurrent build** — fashion-mnist Fused ADC with 6-thread builder: **165 s** (4.9× over serial).

See [`competitive-analysis.md`](competitive-analysis.md) for positioning vs. hnswlib, FAISS, JVector, Lucene, Qdrant and others.

## What's inside

### SIMD kernels (`vectors-core`)

**3–16× faster than scalar Java** for distance computation, via the [Panama Vector API](https://docs.oracle.com/en/java/javase/25/docs/api/jdk.incubator.vector/jdk/incubator/vector/package-summary.html).

- 4× unrolled FMA dot product, L2, cosine similarity
- Conditional FMA dispatch based on hardware capability
- `SPECIES_PREFERRED` auto-adapts to AVX2 / AVX-512 / NEON / SVE
- Byte operations with tiered 512-bit + 256-bit + 128-bit widening
- No JNI, no FFM-to-C++ bindings — pure Java

### Quantization (`vectors-quantization`) — the widest portfolio in the Java ecosystem

| Quantizer | Bits/dim | Reference |
|---|---|---|
| Scalar (int8, int4) | 8, 4 | Standard |
| Product Quantization | ~2–4 | k-means++ per subspace |
| Binary (sign-bit) | 1 | Standard |
| BBQ | 1 + corrections | Lucene-style |
| **RaBitQ** | 1 + corrections | **SIGMOD 2024** |
| **Extended RaBitQ** | 2–8 | **SIGMOD 2025** |
| TurboQuantizer | 1–8 | Lloyd-Max optimal codebooks |
| NVQ | 8 (nonlinear) | arXiv 2509.18471 |

Three pluggable rotation strategies (**Random**, **Givens**, **Quaternion**) for rotation-based quantizers. No other Java library implements the RaBitQ family, pluggable rotations, or anisotropic-PQ (ScaNN-style) training with a coordinate-descent encoder.

### Graph indexing (`vectors-hnsw`, `vectors-vamana`)

- **HNSW** — multi-layer navigable small world. >0.999 recall on SIFT.
- **Vamana** — DiskANN-style single-layer graph with graduated α pruning.
- **Fused ADC** — PQ codes packed inline with the HNSW adjacency list for cache-locality wins on compressed search.
- Both indexes support a **quantized two-pass search**: coarse quantized traversal + full-precision rescore.
- Concurrent graph builder (`ConcurrentHnswGraphBuilder`) with lock-striped neighbor maps.

### Embedded database (`vectors-db`)

- FLAT, HNSW, and Vamana index types
- All quantizer kinds (SQ8, SQ4, PQ, BQ, RaBitQ, NVQ)
- **mmap-persistent** with generation-based atomic commits
- Crash recovery with walk-back
- Tombstone-based deletion with compaction
- Structured metadata storage with filter expressions
- Refcounted read snapshots — **readers never block writers**
- Arrow IPC export

### Framework adapters

- **`vectors-spring-ai`**: `JavaVectorsVectorStore extends AbstractObservationVectorStore` — observability, filters, metadata all wired through.
- **`vectors-langchain4j`**: `JavaVectorsEmbeddingStore implements EmbeddingStore<TextSegment>` — feature parity with `InMemoryEmbeddingStore`.

## When to use java-vectors (and when not to)

| Use case | Recommendation |
|---|---|
| Spring AI / LangChain4j app, ≤10M vectors, single instance | ✅ **Primary fit** |
| Spring Boot service that currently calls out to pgvector / Redis-vector for RAG | ✅ Retire the external database for mid-scale workloads |
| Embedding cache / in-VM reranking beside any LLM pipeline | ✅ Sub-ms recall, portable mmap format |
| Team evaluating JVector outside of Cassandra | ✅ Broader quantization, Spring AI + LangChain4j adapters, HNSW in addition to Vamana |
| JVM application currently using hnswlib / FAISS / USearch via JNI | ✅ Drop the native boundary |
| Graal native-image build with vector search | ✅ No native code we maintain means no FFM/JNI hoops |
| Client-side quantization before a remote DB | ✅ `vectors-quantization` alone |
| Java-native microservice that needs HA without running a separate database | 🛠 `vectors-distributed` embedded cluster — 2026 H2 |
| >100M vectors with GPU budget / latency <5 ms | 🛠 `vectors-gpu` (Panama-FFM + cuVS) — 2026 H2. Until then: FAISS-GPU / Milvus. |
| Need HA replication + multi-tenant RBAC in production **today** | ⏸ Use Qdrant / Weaviate (revisit when `vectors-distributed` GAs) |
| Hybrid BM25 + vector search pipeline | ⏸ Use Vespa / Elasticsearch / Solr |
| Python / Go / Rust application | ⏸ Use hnswlib / USearch / FAISS |

## Modules

| Module | Status | Description |
|---|---|---|
| [vectors-core](vectors-core/) | stable | SIMD distance kernels, vector types, similarity functions |
| [vectors-storage](vectors-storage/) | stable | Off-heap memory, mmap, arena-based storage |
| [vectors-quantization](vectors-quantization/) | stable | SQ, PQ, BQ, RaBitQ, Extended RaBitQ, TurboQuant, NVQ |
| [vectors-hnsw](vectors-hnsw/) | stable | HNSW graph index, concurrent builder, Fused ADC |
| [vectors-vamana](vectors-vamana/) | stable | Vamana / DiskANN graph index |
| [vectors-db](vectors-db/) | stable | Embedded vector database |
| [vectors-spring-ai](vectors-spring-ai/) | stable | Spring AI `VectorStore` adapter |
| [vectors-langchain4j](vectors-langchain4j/) | stable | LangChain4j `EmbeddingStore` adapter |
| [vectors-ivf](vectors-ivf/) | **in progress** | IVF family — BuoyIndex, TieredCluster, SubBuoyTree, HyperDoor, SOAR spilling |
| [vectors-distributed](vectors-distributed/) | **in progress** | Embedded cluster member — gossip (SWIM), consistent-hash sharding, scatter-gather, segment-based state transfer. Patterns from EclipseStore + Eclipse Data Grid, Apache Ignite 3, Hazelcast. |
| vectors-gpu | **planned 2026 H2** | Optional GPU backend via Panama-FFM bindings to NVIDIA cuVS (CAGRA, IVF-PQ, IVF-FLAT). Feature-detected at runtime. |
| [vectors-bench](vectors-bench/) | stable | JMH microbenchmarks + ANN-Benchmarks macrobench harness |

## Dependency graph

```
vectors-core                     <- foundation, no internal deps
vectors-storage                  <- core
vectors-quantization             <- core, storage
vectors-hnsw                     <- core, storage, quantization
vectors-vamana                   <- core, storage, quantization
vectors-ivf                      <- core, storage, quantization        (in progress)
vectors-db                       <- core, storage, quantization, hnsw, vamana
vectors-distributed              <- core, storage, db, ivf             (in progress)
vectors-gpu                      <- core, storage, db                  (planned 2026 H2)
vectors-spring-ai                <- db
vectors-langchain4j              <- db
vectors-bench                    <- all above
```

## How we compare

Compact view. Full matrix, gap analysis, enterprise-distribution patterns, and GPU roadmap in [`competitive-analysis.md`](competitive-analysis.md).

| | **java-vectors** | JVector | Lucene 10.x | hnswlib | FAISS | Qdrant |
|---|---|---|---|---|---|---|
| Pure Java (no JNI we maintain) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Embedded in a Java app | ✅ | ✅ | ✅ | via JNI | via JNI | ❌ (server) |
| HNSW | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Vamana / DiskANN | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| IVF family | 🛠 2026 H1 | ❌ | ❌ | ❌ | ✅ | ◐ |
| Anisotropic PQ (ScaNN-style) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Quantization breadth | **8** | 3 | 3 | 0 | many | 3 |
| SIGMOD 2024 RaBitQ | ✅ | ❌ | partial | ❌ | ❌ | ❌ |
| SIGMOD 2025 Ext-RaBitQ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Pluggable rotations | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Spring AI adapter | ✅ | ❌ | ✅ (ships in Spring AI) | ❌ | ❌ | ✅ |
| LangChain4j adapter | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Embedded cluster (not a server) | 🛠 2026 H2 | ❌ | ❌ | ❌ | ❌ | ❌ |
| Distributed / HA | 🛠 2026 H2 | via Cassandra | via ES/Solr | ❌ | ❌ | ✅ |
| GPU | 🛠 2026 H2 (cuVS) | ❌ | ❌ | ❌ | ✅ | ❌ |
| Min JDK | **25** | 11 | 21 | n/a | n/a | n/a |
| Native deps written by us | **none** | optional JNI | none | JNI | JNI / CUDA | server |

## Requirements

- **JDK 25+** (Vector API incubating + mature FFM)
- **Gradle 9.4+**

The Vector API is wired in automatically via `--add-modules jdk.incubator.vector` on every compile and test task.

## Building

```bash
cd vectors

./gradlew build -x :docs:build   # all library modules
./gradlew test                   # integration + unit tests (excludes slow/benchmark)
./gradlew unitTest               # @Tag("unit") only
./gradlew slowTest               # @Tag("slow") only
./gradlew :vectors-bench:jmh     # JMH microbenchmarks
./gradlew spotlessApply          # Google Java Format 1.35.0
./gradlew :docs:build            # Antora docs site
./gradlew aggregateJavadoc       # combined Javadoc
```

## Roadmap

### 2026 H1 — single-node CPU parity

- `vectors-ivf` IVF_FLAT GA; `vectors-db` integration; IVF_PQ + IVF_PQFS (FastScan) follow-up.
- **Multi-threaded beam search** and **batched query API** — the two biggest lifts toward FAISS-CPU parity.
- **OPQ rotation training** for PQ.
- `IndexRefine` / `IndexShards` / `IndexReplicas` composition wrappers.

### 2026 H2 — enterprise distribution + GPU

- `vectors-distributed` GA — embedded cluster members (EclipseStore + Eclipse Data Grid pattern): SWIM gossip for liveness, Raft for metadata, owned-partition replicas with versioned anti-entropy (Hazelcast pattern), segment-based state transfer (Infinispan pattern), `CacheMode` switch (LOCAL / REPL / DIST), pluggable `DiscoveryStrategy` SPI (multicast / static / Kubernetes / Consul), OpenTelemetry + Micrometer metrics.
- `vectors-gpu` MVP — Panama-FFM bindings to NVIDIA cuVS (CAGRA, IVF-PQ, IVF-FLAT, Brute Force). Runtime-gated on `libcuvs.so` presence. Same public API; transparent fallback to CPU when GPU is absent.
- Optional `vectors-grpc` front-end for polyglot clients.

### 2027+ — native-Java GPU

- Prototype kernels via **Project Babylon + HAT** (`@CodeReflection` + `NDRange`) on JDK 26+ EA builds. Target: cosine/L2 on int8 matching ~90% of cuVS without any native binding.

### Ongoing

- Anisotropic PQ refinements, NVQ scoring kernels, binary-graph indexes, rolling-upgrade catalog versioning (Ignite 3 pattern), multi-tenant RBAC (Infinispan `AuthorizationManager` pattern).

## Further reading

- [`competitive-analysis.md`](competitive-analysis.md) — full market positioning, feature matrices, FAISS gap-closers (§4), enterprise distribution architecture (§11), GPU roadmap (§12).
- [`vectors-distributed-design.md`](vectors-distributed-design.md) — distributed DB design draft.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — engineering plan and phase tracking.

## License

The core libraries, adapters, caching, benchmarks, and demos are licensed under the [Apache License 2.0](LICENSE). The `vectors-distributed`, `vectors-server`, and `vectors-gpu` modules are licensed under the [Functional Source License 1.1 (FSL-1.1-ALv2)](LICENSE-FSL), which automatically converts to Apache 2.0 on April 25, 2028.

See [LICENSING.md](LICENSING.md) for the full module classification and details.
