# java-vectors

**The SIMD-native vector search engine for Java.** Pure Java. Zero dependencies. JDK 25+.

java-vectors provides two things the Java AI ecosystem is missing:

1. **SIMD-accelerated vector operations** -- the numerical foundation that neither Spring AI nor LangChain4j has built. 3-16x faster distance computation than scalar Java, via the [Panama Vector API](https://docs.oracle.com/en/java/javase/25/docs/api/jdk.incubator.vector/jdk/incubator/vector/package-summary.html).

2. **An embedded vector database** -- production-grade HNSW/Vamana indexing, 7 quantization algorithms (including SIGMOD 2024/2025 state of the art), mmap persistence, sub-millisecond queries. Drop it into Spring AI or LangChain4j as a replacement for SimpleVectorStore / InMemoryEmbeddingStore -- no external infrastructure, no network hop, no cost.

## Why java-vectors?

### The Problem

Spring AI's `SimpleVectorStore` and LangChain4j's `InMemoryEmbeddingStore` are the only zero-dependency vector stores in the Java ecosystem. Both are explicitly "not for production":

| | SimpleVectorStore | InMemoryEmbeddingStore |
|--|-------------------|------------------------|
| **Search** | O(n) brute-force | O(n) brute-force |
| **SIMD** | None (scalar loops) | None (scalar loops) |
| **Indexing** | None | None |
| **Quantization** | None | None |
| **Persistence** | JSON file | JSON file |
| **Distance metrics** | Cosine only | Cosine only |
| **Practical limit** | ~10-50K vectors | ~10-50K vectors |

The next step is provisioning an external vector database (Pinecone, Redis, pgvector, etc.) -- adding infrastructure complexity, network latency, and $50-300+/month in hosting costs.

### The Solution

java-vectors fills the gap between "toy in-memory store" and "deploy an external database":

```java
// Spring AI -- drop-in replacement for SimpleVectorStore
VectorCollection collection = VectorCollection.builder()
    .dimension(embeddingModel.dimensions())
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .quantizer(QuantizerKind.SQ8)
    .build();

VectorStore store = JavaVectorsVectorStore.builder(embeddingModel, collection)
    .build();

// Same VectorStore interface, same Spring AI code -- now with HNSW indexing,
// SIMD-accelerated search, scalar quantization, and mmap persistence.
```

```java
// LangChain4j -- drop-in replacement for InMemoryEmbeddingStore
VectorCollection collection = VectorCollection.builder()
    .dimension(384)
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.HNSW)
    .build();

EmbeddingStore<TextSegment> store = JavaVectorsEmbeddingStore.builder(collection)
    .build();
```

At 100K vectors with 1536 dimensions:

| Metric | SimpleVectorStore | java-vectors |
|--------|-------------------|-------------|
| Query latency | ~20-170ms | <1ms |
| Memory | ~586 MB (float32 on-heap) | ~147 MB (SQ8, off-heap) |
| Recall | 100% (brute-force) | >99% (HNSW) |
| Persistence | JSON (slow, huge) | mmap (instant, compact) |

## Use It at Any Layer

java-vectors is modular. Use only what you need:

| Use Case | What You Need |
|----------|---------------|
| SIMD distance computation for any Java app | `vectors-core` |
| Quantize embeddings before sending to Redis/Qdrant/Milvus | `vectors-core` + `vectors-quantization` |
| Build a custom ANN index | `vectors-core` + `vectors-quantization` + `vectors-hnsw` or `vectors-vamana` |
| Embedded vector database | `vectors-db` (pulls in everything) |
| Spring AI drop-in replacement | `vectors-spring-ai` |
| LangChain4j drop-in replacement | `vectors-langchain4j` |

### Client-Side Quantization

Even if you use a remote vector database, java-vectors adds value. 5 of 8 major vector databases accept pre-quantized vectors from clients:

```java
// Quantize embeddings client-side before sending to your remote DB
ScalarQuantizer sq = ScalarQuantizer.train(dataset, ScalarBits.INT8);
byte[] quantized = sq.encode(floatVector);
// Send to Redis, Qdrant, Milvus, Elasticsearch, or pgvector
// -> 75% bandwidth reduction, lower storage costs
```

| Quantization | Compression | Recall Impact | Network Savings |
|-------------|-------------|---------------|-----------------|
| Scalar int8 | 4x | ~1.5% recall drop | 75% |
| Product Quantization | 10-90x | 2-5% recall drop | 90%+ |
| Binary (1-bit) | 32x | 5-10% (before rescore) | 97% |
| RaBitQ (1-bit + corrections) | 32x + corrections | <2% (after rescore) | ~90% |

## What's Inside

### SIMD Kernels (`vectors-core`)

3-16x faster than scalar Java for vector distance computation:

- 4x unrolled FMA dot product, L2, cosine similarity
- Conditional FMA dispatch based on hardware capability
- `SPECIES_PREFERRED` for automatic adaptation to AVX2, AVX-512, NEON, SVE
- Byte operations with tiered 512-bit + 256-bit widening
- No JNI, no FFM-to-C++ bindings -- pure Java Vector API

### Quantization (`vectors-quantization`)

The most comprehensive quantization portfolio in the Java ecosystem:

| Quantizer | Bits/dim | Source |
|-----------|----------|--------|
| Scalar (int8, int4) | 8, 4 | Standard |
| Product Quantization | ~2-4 | k-means++ per subspace |
| Binary (sign-bit) | 1 | Standard |
| BBQ | 1 + corrections | Lucene-style |
| RaBitQ | 1 + corrections | SIGMOD 2024 |
| Extended RaBitQ | 2-8 | SIGMOD 2025 |
| TurboQuantizer | 1-8 | Lloyd-Max optimal codebooks |
| NVQ | 8 (nonlinear) | arXiv 2509.18471 |

Three pluggable rotation strategies (Random, Givens, Quaternion) for rotation-based quantizers. No other Java library implements the RaBitQ family or pluggable rotations.

### Graph Indexing (`vectors-hnsw`, `vectors-vamana`)

- **HNSW**: Multi-layer navigable small world graph. >0.999 recall on SIFT.
- **Vamana**: DiskANN-style single-layer graph with graduated alpha pruning.
- Both support quantized two-pass search (coarse quantized pass + full-precision rescore).

### Embedded Database (`vectors-db`)

- FLAT, HNSW, and VAMANA index types
- All quantizer kinds (SQ8, SQ4, PQ, BQ, RABITQ, NVQ)
- mmap-persistent with generation-based atomic commits
- Crash recovery with walk-back
- Tombstone-based deletion with compact
- Metadata storage with filter expressions
- Refcounted read snapshots (readers never block writers)

### Framework Adapters

- **`vectors-spring-ai`**: `JavaVectorsVectorStore extends AbstractObservationVectorStore`
- **`vectors-langchain4j`**: `JavaVectorsEmbeddingStore implements EmbeddingStore<TextSegment>`

## Modules

| Module | Description |
|--------|-------------|
| [vectors-core](vectors-core/) | SIMD distance kernels, vector types, similarity functions |
| [vectors-storage](vectors-storage/) | Off-heap memory, mmap, arena-based storage |
| [vectors-quantization](vectors-quantization/) | SQ, PQ, BQ, RaBitQ, Extended RaBitQ, TurboQuant, NVQ |
| [vectors-hnsw](vectors-hnsw/) | HNSW graph index |
| [vectors-vamana](vectors-vamana/) | Vamana/DiskANN graph index |
| [vectors-db](vectors-db/) | Embedded vector database |
| [vectors-spring-ai](vectors-spring-ai/) | Spring AI VectorStore adapter |
| [vectors-langchain4j](vectors-langchain4j/) | LangChain4j EmbeddingStore adapter |
| [vectors-ivf](vectors-ivf/) | IVF family indexes — BuoyIndex, TieredCluster, SOAR spilling (planned) |
| [vectors-distributed](vectors-distributed/) | Distributed vector database — gossip routing, scatter-gather, S3 WAL (planned) |
| [vectors-bench](vectors-bench/) | JMH benchmarks and ANN-Benchmarks harness |

## Dependency Graph

```
vectors-core                     <- foundation, no internal deps
vectors-storage                  <- core
vectors-quantization             <- core, storage
vectors-hnsw                     <- core, storage, quantization
vectors-vamana                   <- core, storage, quantization
vectors-ivf                      <- core, storage, quantization           (planned)
vectors-db                       <- core, storage, quantization, hnsw, vamana
vectors-distributed              <- core, storage, db, ivf               (planned)
vectors-spring-ai                <- db
vectors-langchain4j              <- db
vectors-bench                    <- all above
```

## Compared to Alternatives

| Feature | java-vectors | JVector (DataStax) | Apache Lucene 10.x |
|---------|-------------|-------------------|-------------------|
| Graph Algorithms | HNSW + Vamana | Vamana only | HNSW only |
| Quantization Families | 8 | 3 (PQ, BQ, NVQ) | 3 (SQ, multi-bit, BBQ) |
| RaBitQ (SIGMOD 2024) | Yes | No | Partial (BBQ) |
| Extended RaBitQ (SIGMOD 2025) | Yes | No | No |
| Pluggable Rotations | Yes | No | No |
| Embedded Database | Yes | Via EclipseStore 4 | Via Elasticsearch/Solr |
| Framework Adapters | Spring AI, LangChain4j | None | None |
| Min JDK | 25 | 11 | 21 |
| Native Dependencies | None | Optional JNI | None |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |

## Requirements

- **JDK 25+**
- **Gradle 9.4+**

## Building

```bash
cd vectors

# Build all library modules
./gradlew build -x :docs:build

# Run tests
./gradlew test

# Unit tests only
./gradlew unitTest

# Code formatting
./gradlew spotlessApply

# Documentation
./gradlew :docs:build
./gradlew aggregateJavadoc
```

The Vector API is added automatically via `--add-modules jdk.incubator.vector` on compile and test tasks.

## License

Apache License 2.0
