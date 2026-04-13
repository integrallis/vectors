# Distributed Vector Database: Design and Development Plan

**Status**: Design Draft (April 2026)
**Author**: java-vectors project
**Inputs**: `tiered-buoy-architecture.md`, `distributed-vector-search-analysis.md`, `design-strategy.md`

---

## 1. Purpose

This document is the authoritative engineering plan for evolving java-vectors from an
embedded single-node vector database into a distributed vector database. It follows the
project's TDD discipline: **acceptance tests and JMH benchmark targets are specified before
any implementation line is written**. Every section that introduces a new type specifies its
test class and its benchmark first.

The full distributed implementation lives in a new `vectors-distributed` module. The
existing modules — `vectors-core`, `vectors-storage`, `vectors-db`, `vectors-quantization`,
`vectors-hnsw`, `vectors-vamana` — receive **minimal, targeted primitive additions** that
make them distribution-aware without coupling them to a particular distribution strategy.
A new `vectors-ivf` module provides the IVF routing layer. Distribution is a layer on top.

### 1.1 Non-Goals

- No changes to the `VectorCollection` public API as seen by Spring AI / LangChain4j adapters
- No consensus protocol (Raft/Paxos) — S3 compare-and-swap is the coordination primitive
- No JNI, no native bindings, no C++ dependencies — pure Java throughout
- `vectors-distributed` does **not** depend on `vectors-spring-ai` or `vectors-langchain4j`

---

## 2. Design Principles

1. **Embedded first** — the single-node `VectorCollection` must work identically at every
   scale. The distributed layer is a thin shell, not a redesign.
2. **Quantization = tier** — 1-bit RaBitQ codes live on every node (globally replicated);
   full-precision vectors live only where they're needed (SSD/S3).
3. **S3 as source of truth** — no replication protocol. Write durability = S3 PutObject.
4. **Virtual threads for scatter-gather** — Java Loom eliminates thread-pool tuning.
5. **Test-first** — each primitive has an acceptance test before the first implementation line.
6. **Benchmark-gated** — each phase has explicit JMH throughput/latency targets that must
   pass before the phase is considered complete.

---

## 3. Architecture Overview

The distributed architecture has four layers, each buildable and testable independently:

```
Layer 3: vectors-distributed   DistributedVectorCollection, ShardRouter,
                               ScatterGatherExecutor, GossipMembership,
                               S3WriteAheadLog, EventLog

Layer 2: vectors-ivf           BuoyIndex (centroid routing), IvfIndex,
                               ClusterPartition, SoarSpillMap,
                               TieredCluster (hyper doors, multi-pass cascade)

Layer 1: vectors-db            VectorCollection (unchanged public API),
         vectors-storage       StorageBackend interface (new),
         vectors-core          KMeans, CentroidIndex, batch SIMD distance (new)

Layer 0: vectors-core          SIMD kernels (unchanged)
         vectors-quantization  Quantizers (unchanged, consumed by tiers)
```

The query path in distributed mode:

```
Client query
  |
  v
ShardRouter.route(query, nprobe)       -- BuoyIndex: K=1024 SIMD centroids, <5 µs
  |
  v
ScatterGatherExecutor (virtual threads)
  |-- Node A: localSearch(clusterId=5,  query, k) -- HNSW within cluster, <2 ms
  |-- Node B: localSearch(clusterId=12, query, k) -- HNSW within cluster, <2 ms
  |-- Node C: localSearch(clusterId=47, query, k) -- HNSW within cluster, <2 ms
  |
  v
TopKMerger.merge(results, k)           -- heap merge, O(n_shards * k * log k)
  |
  v
Client: top-k results
```

---

## 4. Module Impact Summary

| Module | Change type | What changes |
|--------|-------------|--------------|
| `vectors-core` | Additions | `KMeans`, `CentroidIndex`, batch SIMD distance methods in `VectorUtil` |
| `vectors-storage` | Additions | `StorageBackend` interface, `LocalStorageBackend`, `WriteAheadLog` |
| `vectors-db` | Additions | `VectorEvent` sealed hierarchy, `EventLog`, `SegmentExporter/Importer` |
| `vectors-quantization` | No change | Consumed as-is by `TieredCluster` |
| `vectors-hnsw` | No change | `HnswGraph.merge()` deferred to Phase 4 |
| `vectors-vamana` | No change | Used within clusters |
| `vectors-ivf` | **New module** | `BuoyIndex`, `IvfIndex`, `ClusterPartition`, `SoarSpillMap`, `TieredCluster` |
| `vectors-distributed` | **New module** | `DistributedVectorCollection`, `ShardRouter`, `ScatterGatherExecutor`, gossip, WAL |
| `vectors-bench` | Additions | IVF + distributed benchmark classes |

---


## 5. Primitive Additions — `vectors-core`

### 5.1 Batch SIMD Distance (additions to `VectorUtil`)

IVF centroid routing requires comparing one query against K centroids in a tight loop.
The existing `VectorUtil` scalar-pair methods are per-pair; these new methods operate
over an entire centroid matrix in a single SIMD-batched pass.

**Test class**: `VectorUtilBatchTest` (new, `@Tag("unit")`)

```java
// Fills out[i] = dot(query, matrix[i]) for i in [0, k)
static void batchDotProduct(float[] query, float[][] matrix, float[] out);
// Fills out[i] = squaredL2(query, matrix[i])
static void batchSquaredL2(float[] query, float[][] matrix, float[] out);
```

Acceptance tests (must pass before any `CentroidIndex` or `BuoyIndex` code is written):
- `batchDotProduct_matchesScalarLoop` — batch vs per-pair, K=1024 D=128, tolerance 1e-5
- `batchSquaredL2_matchesScalarLoop` — same agreement
- `batchDotProduct_handlesKNotMultipleOf4` — K=1000
- `batchDotProduct_handlesK1` — degenerate K=1 edge case

**Benchmark class**: `VectorUtilBatchBenchmark` (new, `vectors-bench`)

| Benchmark | K | D | Target |
|-----------|---|---|--------|
| `batchDotProduct_K1024_D128` | 1024 | 128 | ≥ 2 GB/s effective data throughput |
| `batchSquaredL2_K1024_D128` | 1024 | 128 | ≥ 2 GB/s effective data throughput |
| `batchDotProduct_K16384_D768` | 16384 | 768 | ≥ 1 GB/s (billion-scale routing) |

### 5.2 `KMeans` — SIMD-accelerated centroid training

Package: `com.integrallis.vectors.core.cluster`

```java
public final class KMeans {
    /**
     * Train k centroids from a dataset sample using Lloyd's algorithm with k-means++
     * seeding. Uses batchSquaredL2 for assignment. Parallelises centroid-update
     * (averaging) over virtual threads when dataset exceeds 100K rows.
     *
     * @param dataset   row-major float matrix [n][dim]
     * @param k         centroid count; ceil(sqrt(n)) is the FAISS guideline
     * @param maxIter   Lloyd iterations (100 default)
     * @param seed      RNG seed for reproducibility
     * @return centroid matrix [k][dim]
     */
    public static float[][] train(float[][] dataset, int k, int maxIter, long seed);

    /** Assign each row in dataset to its nearest centroid. Returns int[n]. */
    public static int[] assign(float[][] dataset, float[][] centroids);

    /** Return the within-cluster sum of squared distances (quantisation error). */
    public static double quantizationError(float[][] dataset, float[][] centroids,
                                            int[] assignments);
}
```

**Test class**: `KMeansTest` (new, `@Tag("unit")`)

Acceptance tests:
- `twoWellSeparatedGaussiansRecoversCentroids` — |recovered - true|² < 0.01 per dim
- `assignAllPointsToNearestCentroid` — post-train assignment matches brute-force argmin
- `trainConvergesWithinMaxIter` — no more iterations than maxIter parameter
- `sameSeedException` — two calls, same seed → bit-identical centroids
- `kEqualsOne` — degenerate case, single centroid = dataset mean
- `kEqualsN` — each centroid is one vector (trivial partition)
- `trainWithVirtualDataset_N1M_K1024_D128` — tagged `@Tag("slow")`, verifies recall
  improvement (SIFT-1M recall@10 with nprobe=32 ≥ 0.95 after training)

**Benchmark class**: `KMeansBenchmark` (`vectors-bench`, `@BenchmarkMode(Mode.SingleShotTime)`)

| Benchmark | N | K | D | Target |
|-----------|---|---|---|--------|
| `train_N100K_K1024_D128` | 100K | 1024 | 128 | < 5 s |
| `train_N1M_K1024_D128` | 1M | 1024 | 128 | < 120 s (`@Tag("slow")`) |
| `assign_N1M_K1024_D128` | 1M | 1024 | 128 | < 10 s |

### 5.3 `CentroidIndex` — multi-probe centroid routing

Package: `com.integrallis.vectors.core.cluster`

```java
public final class CentroidIndex {
    public CentroidIndex(float[][] centroids, SimilarityFunction metric);

    /** Top-nprobe centroid ids sorted by ascending distance to query. */
    public int[] route(float[] query, int nprobe);

    /**
     * SOAR-style boundary expansion. Any centroid whose distance is within
     * (1 + gamma) * nearestDistance gets its spillTarget added to the result set.
     * Result is deduplicated and sorted by ascending distance.
     */
    public int[] routeWithSpill(float[] query, int nprobe, float gamma, int[] spillTargets);

    public int centroidCount();
    public int dimension();
}
```

**Test class**: `CentroidIndexTest` (new, `@Tag("unit")`)

Acceptance tests:
- `routeReturnsNprobeIds` — result.length == nprobe (unless K < nprobe)
- `routeResultsSortedByAscendingDistance` — verified by computing distances after
- `routeWithSpill_addsSpillTargetWhenBoundary` — gamma=0.2, confirm expansion triggered
- `routeWithSpill_noDuplicates` — even when spill target == primary target
- `routeWithSpill_noExpansionWhenFarFromBoundary` — tight gamma=0.0 adds nothing

**Benchmark class**: `CentroidIndexBenchmark` (`vectors-bench`, throughput mode)

| Benchmark | K | D | nprobe | Target |
|-----------|---|---|--------|--------|
| `route_K1024_D128_nprobe32` | 1024 | 128 | 32 | < 10 µs/call |
| `routeWithSpill_K1024_D128` | 1024 | 128 | 32 | < 15 µs/call |
| `route_K16384_D768_nprobe64` | 16384 | 768 | 64 | < 200 µs/call |

---

## 6. Primitive Additions — `vectors-storage`

### 6.1 `StorageBackend` — pluggable persistence

Package: `com.integrallis.vectors.storage.backend`

```java
public interface StorageBackend extends Closeable {
    void put(String key, byte[] data) throws IOException;
    byte[] get(String key) throws IOException;        // null if absent
    boolean exists(String key) throws IOException;
    void delete(String key) throws IOException;
    List<String> list(String prefix) throws IOException;

    /**
     * Atomic conditional put (compare-and-swap on etag). Returns succeeded=true
     * and newEtag when the stored etag matches expectedEtag (or key is absent and
     * expectedEtag is null). Returns succeeded=false on mismatch.
     * Implementations that do not support CAS throw UnsupportedOperationException.
     */
    ConditionalPutResult conditionalPut(String key, byte[] data, String expectedEtag)
        throws IOException;

    record ConditionalPutResult(boolean succeeded, String newEtag) {}
}
```

Implementations (Phase 1):
- `HeapStorageBackend` — `ConcurrentHashMap<String,byte[]>`, CRC-less, for tests
- `LocalFileStorageBackend` — wraps the existing file layout (file path = key)
- `S3StorageBackend` — lives in `vectors-distributed` (Phase 3, requires AWS SDK)

**Test class**: `StorageBackendContractTest` (new, `@ParameterizedTest` over both impls)

Acceptance tests (same body for all implementations):
- `putAndGetRoundTrip`
- `getMissingKeyReturnsNull`
- `listReturnsKeysUnderPrefix`
- `deleteRemovesKey`
- `conditionalPut_succeedsWhenKeyAbsent` — expectedEtag=null
- `conditionalPut_failsOnStaleEtag` — returns succeeded=false, data unchanged
- `conditionalPut_succeedsWithCurrentEtag` — sequential chain of CAS succeeds

### 6.2 `WriteAheadLog` — sequence-numbered durable journal

Package: `com.integrallis.vectors.storage.wal`

```java
public interface WriteAheadLog extends Closeable {
    long append(byte[] entry) throws IOException;          // returns seq number
    Stream<WalEntry> readFrom(long fromSeqInclusive) throws IOException;
    long lastSequenceNumber();                             // -1 if empty
    void flush() throws IOException;

    record WalEntry(long sequenceNumber, byte[] data) {}
}
```

Implementation: `SegmentedWriteAheadLog` — entries stored in fixed-size segment files
(default 64 MB) on any `StorageBackend`. Each entry has a 4-byte length prefix + 4-byte
CRC32. A closed segment is immutable. Active segment accepts appends.

**Test class**: `WriteAheadLogTest` (new, `@Tag("unit")`)

Acceptance tests:
- `appendThenReadRecoversEntry`
- `sequenceNumbersStrictlyMonotonic`
- `readFromZeroReturnsAll`
- `readFromMidSeqSkipsEarlierEntries`
- `segmentRolloverCreatesNewSegment` — write > 64 MB, verify two segment files appear
- `crcCorruptionDetectedOnRead` — flip a byte in a closed segment → IOException
- `emptyLogReturnsMinusOneSequence`

**Benchmark class**: `WriteAheadLogBenchmark` (`vectors-bench`)

| Benchmark | Entry size | Backend | Target |
|-----------|-----------|---------|--------|
| `append_1KB` | 1 KB | LocalFile | ≥ 50K appends/s |
| `append_64KB` | 64 KB | LocalFile | ≥ 5K appends/s |
| `scan_100K_entries` | 1 KB | LocalFile | ≥ 500K entries/s |

---


## 7. Primitive Additions — `vectors-db`

### 7.1 `VectorEvent` — the mutation event hierarchy

The event-sourced mutation model. Every change to a collection is recorded as an event.
The event log is the replication and recovery primitive for the distributed layer.
Package: `com.integrallis.vectors.db.event`

```java
/**
 * A sealed mutation event emitted by VectorCollection operations.
 * Every event carries the collection id (for multi-tenant logs), a sequence number
 * (assigned by the EventLog), and the epoch millis at which it was created.
 */
public sealed interface VectorEvent
    permits VectorEvent.Add, VectorEvent.Delete, VectorEvent.Upsert,
            VectorEvent.MetadataUpdate {

    String collectionId();

    record Add(String collectionId, String documentId, float[] vector,
               String text, Map<String, MetadataValue> metadata)
        implements VectorEvent {}

    record Delete(String collectionId, String documentId)
        implements VectorEvent {}

    record Upsert(String collectionId, String documentId, float[] vector,
                  String text, Map<String, MetadataValue> metadata)
        implements VectorEvent {}

    record MetadataUpdate(String collectionId, String documentId,
                          Map<String, MetadataValue> metadata)
        implements VectorEvent {}
}
```

`VectorEvent` objects are serialised to `byte[]` by `VectorEventCodec` (a simple binary
format: type tag + varint lengths + raw field bytes + CRC32). This codec is used by both
the local WAL and the distributed S3 WAL.

**Test class**: `VectorEventCodecTest` (new, `@Tag("unit")`)

Acceptance tests (for each event type):
- `addEventRoundTrips` — serialise then deserialise, all fields preserved
- `deleteEventRoundTrips`
- `upsertEventRoundTrips`
- `metadataUpdateEventRoundTrips`
- `crcMismatchRejected` — corrupt a byte → IOException
- `unknownTypeTagRejected` — invalid type byte → IOException

### 7.2 `SegmentExporter` / `SegmentImporter` — portable generation snapshots

Enables a node to export its current committed state and import it on another node,
bootstrapping a replica without replaying the full event log.
Package: `com.integrallis.vectors.db.transfer`

```java
/** Exports the current committed generation to a single, self-describing byte stream. */
public interface SegmentExporter {
    /**
     * Writes a portable snapshot of the current generation to out.
     * Format: manifest header + vectors.bin + metadata.bin + idmap.bin +
     *         (optional) graph.bin + (optional) tombstones.bin
     * All fields are length-prefixed with CRC32.
     * Thread-safe: reads from an immutable generation snapshot.
     */
    void export(VectorCollection collection, OutputStream out) throws IOException;
}

/** Imports a snapshot into a fresh VectorCollection backed by the given storage root. */
public interface SegmentImporter {
    VectorCollection importSnapshot(InputStream in, Path storageRoot) throws IOException;
}
```

**Test class**: `SegmentTransferTest` (new, `@Tag("unit")`)

Acceptance tests:
- `exportThenImport_flatCollection` — search results identical before and after transfer
- `exportThenImport_hnswCollection` — graph structure preserved, search results match
- `exportThenImport_withTombstones` — tombstones preserved
- `exportThenImport_emptyCollection` — zero documents, no error
- `importRejectsCorruptMagic` — bad header bytes → IOException
- `importRejectsChecksumMismatch` — corrupt vector bytes → IOException

**Benchmark class**: `SegmentTransferBenchmark` (`vectors-bench`)

| Benchmark | N | D | Index | Target |
|-----------|---|---|-------|--------|
| `export_N100K_D128_FLAT` | 100K | 128 | FLAT | < 2 s |
| `export_N100K_D128_HNSW` | 100K | 128 | HNSW | < 3 s |
| `import_N100K_D128_FLAT` | 100K | 128 | FLAT | < 3 s |

---

## 8. New Module — `vectors-ivf`

### 8.1 Purpose

The IVF layer sits between `vectors-core` (distance kernels, k-means) and
`vectors-distributed` (multi-node routing). It implements:
- The `BuoyIndex` (global centroid routing index)
- `ClusterPartition` (per-centroid vector posting list)
- `IvfIndex` (flat IVF: route + brute-force within cluster)
- `IvfHnswIndex` (route + HNSW within cluster, Phase 2)
- `TieredCluster` (per-cluster hot/warm/cold tiers, Phase 2)
- SOAR spill map construction

Dependencies: `vectors-core`, `vectors-storage`, `vectors-quantization`

### 8.2 `BuoyIndex`

Package: `com.integrallis.vectors.ivf`

```java
/**
 * The global centroid routing index. Wraps a CentroidIndex with an optional
 * SOAR spill map and cluster metadata (sizes, radii for boundary detection).
 * Always resident in JVM heap. Immutable once built; rebuilt on retrain.
 */
public final class BuoyIndex {
    /**
     * Build a BuoyIndex by training k-means over a representative sample of dataset.
     * Constructs the spill map by finding the second-nearest centroid for each
     * training point, per the SOAR construction algorithm.
     *
     * @param dataset    training sample, shape [n][dim]; n >= 256*k recommended
     * @param k          number of buoys (centroids)
     * @param metric     similarity function
     * @param buildSoar  whether to compute SOAR spill targets (recommended: true)
     * @param seed       RNG seed
     */
    public static BuoyIndex train(float[][] dataset, int k, SimilarityFunction metric,
                                   boolean buildSoar, long seed);

    /** Route query to the nprobe nearest clusters, with optional SOAR expansion. */
    public int[] route(float[] query, int nprobe, float gamma);

    public int k();
    public SimilarityFunction metric();
    public float[][] buoyVectors();

    /** Encode to bytes for persistence/gossip. */
    public byte[] encode();
    public static BuoyIndex decode(byte[] bytes);
}
```

**Test class**: `BuoyIndexTest` (new, `@Tag("unit")`)

Acceptance tests:
- `trainProducesKBuoys` — buoyVectors().length == k
- `routeReturnsNprobeClusterIds`
- `soarSpillMapReducesBoundaryMisses` — synthetic boundary query, spill-augmented routing
  retrieves the true nearest neighbor that plain nprobe routing misses
- `encodeDecodeRoundTrip` — encode then decode, all fields bit-identical
- `routeIsIdempotent` — same query twice → same result (no mutating state)

**Benchmark class**: `BuoyIndexBenchmark` (`vectors-bench`)

| Benchmark | K | D | nprobe | Target |
|-----------|---|---|--------|--------|
| `route_K1024_D128_nprobe32` | 1024 | 128 | 32 | < 10 µs/call |
| `route_K1024_D128_nprobe64_soar` | 1024 | 128 | 64 | < 20 µs/call |

### 8.3 `ClusterPartition`

```java
/**
 * A single IVF cluster: the centroid + an ordered posting list of document ordinals
 * within that cluster. Immutable. Backed by an int[] ordinals array.
 */
public record ClusterPartition(
    int clusterId,
    float[] centroid,
    int[] ordinals,          // global ordinals into the parent VectorCollection
    int size                 // ordinals.length
) {
    public boolean isEmpty() { return size == 0; }
}
```

### 8.4 `IvfIndex` — flat IVF (Phase 1 target)

```java
/**
 * IVF-flat index: route query to nprobe clusters via BuoyIndex, then brute-force
 * scan each cluster's vectors. The simplest correct IVF implementation.
 */
public final class IvfIndex implements Closeable {
    public static IvfIndex build(VectorCollection collection, IvfBuildParams params);

    /**
     * Search: route → brute-force per cluster → merge top-k.
     * Total candidates = nprobe * cluster_size; brute-force within each.
     */
    public SearchResult search(IvfSearchRequest request);

    public BuoyIndex buoyIndex();
    public int clusterCount();
    public ClusterPartition partition(int clusterId);
}

public record IvfBuildParams(
    int k,           // number of clusters (ceil(sqrt(n)) recommended)
    int maxIter,     // k-means iterations
    float gamma,     // SOAR boundary expansion (0.0 = disabled)
    boolean buildSoar,
    long seed
) {}

public record IvfSearchRequest(
    float[] query,
    int k,             // top-k results
    int nprobe,        // clusters to probe
    float gamma,       // SOAR expansion at search time (0.0 = disabled)
    float minScore     // score threshold (default -Float.MAX_VALUE)
) {}
```

**Integration test class**: `IvfIndexIntegrationTest` (new, `@Tag("unit")`)

Acceptance tests — these are the Phase 1 gate:
- `buildAndSearchFlat_10K_K64` — build on 10K 128-d vectors, search 100 queries,
  recall@10 ≥ 0.90 with nprobe=8 (vs. brute-force ground truth)
- `buildAndSearchFlat_100K_K316_nprobe32` — recall@10 ≥ 0.95 (`@Tag("slow")`)
- `searchReturnsAtMostKResults`
- `searchWithEmptyCollectionReturnsEmpty`
- `nprobeEqualsKGivesBruteForceRecall` — nprobe=k → recall@10 ≥ 0.999
- `soarExpansionImprovesRecallAtBoundary` — gamma=0.2 recall ≥ plain nprobe recall

**Benchmark class**: `IvfIndexBenchmark` (`vectors-bench`, `@Tag("slow")`)

| Benchmark | N | K | D | nprobe | Target |
|-----------|---|---|---|--------|--------|
| `search_N1M_K1024_D128_nprobe32` | 1M | 1024 | 128 | 32 | < 5 ms p99 |
| `search_N1M_K1024_D128_nprobe64` | 1M | 1024 | 128 | 64 | < 8 ms p99 |
| `search_N100M_K10000_D128_nprobe64` | 100M | 10000 | 128 | 64 | < 20 ms p99 |

### 8.5 `TieredCluster` — per-cluster multi-pass cascade (Phase 2)

Encapsulates the hot/warm/cool tier model within a single cluster. The cascade:

1. **T0 (1-bit RaBitQ heap)**: Score ALL ordinals → retain top-`t0Candidates`
2. **T1 (SQ8 off-heap)**: Rescore survivors → retain top-`t1Candidates`
3. **T2 (float32 mmap)**: Exact rescore → return top-k

```java
public final class TieredCluster {
    int clusterId();
    int size();

    /**
     * Multi-pass cascade search. Returns top-k scored results.
     * Tiers not yet materialised fall back to the next coarser tier.
     */
    SearchResult search(float[] query, int k, CascadeParams params);

    /** Demand-materialise T1 (SQ8) codes from the provided source. */
    void materializeT1(SQ8ClusterSource source);

    /** Demand-materialise T2 (float32) via mmap of the cluster's vectors.bin. */
    void materializeT2(Path clusterVectorsFile);

    /** Evict T1 to free off-heap memory. T0 and T2 are unaffected. */
    void evictT1();

    record CascadeParams(int t0Candidates, int t1Candidates) {}
}
```

---


## 9. New Module — `vectors-distributed`

### 9.1 Purpose and Boundaries

`vectors-distributed` wires together the IVF routing layer with a multi-node execution
fabric. It is the only module that:
- Knows about nodes, network addresses, and gossip
- Owns the S3 write-ahead log and `S3StorageBackend`
- Implements `DistributedVectorCollection`, which honours the full `VectorCollection` API

Everything in this module has a pure-Java in-process simulation so that tests never
require real network or S3 access.

Dependencies: `vectors-core`, `vectors-storage`, `vectors-db`, `vectors-ivf`
Optional (compile-only): AWS SDK v2 (`software.amazon.awssdk:s3`), for `S3StorageBackend`

### 9.2 Core Types

Package: `com.integrallis.vectors.distributed`

```java
/** Opaque identifier for a cluster node. Immutable value type. */
public record NodeId(String id) {}

/** The ownershipmap: for each cluster id, which node is the primary. */
public interface ShardOwnership {
    NodeId primaryFor(int clusterId);
    List<NodeId> replicasFor(int clusterId);      // Phase 3
    Set<Integer> clustersFor(NodeId node);
    int totalClusters();
}

/** Static shard ownership computed from BuoyIndex + consistent hashing. */
public final class ConsistentHashShardOwnership implements ShardOwnership { ... }

/** Routes IvfSearchRequests to the correct node(s) and collects results. */
public interface ShardRouter {
    /**
     * Given a query's cluster ids (from BuoyIndex.route), return one
     * LocalSearchRequest per node that owns at least one of those clusters.
     * Merges cluster ids for nodes that own multiple.
     */
    List<LocalSearchRequest> plan(float[] query, int[] clusterIds, int k);
}

/** A search request destined for a single node. */
public record LocalSearchRequest(
    NodeId targetNode,
    float[] query,
    int[] clusterIds,   // clusters this node should search
    int k,
    float minScore
) {}
```

### 9.3 `ScatterGatherExecutor`

The execution engine. Uses virtual threads (one per node per query) to fan out
`LocalSearchRequest`s, collect results, and merge them.

```java
/**
 * Stateless scatter-gather over a NodeDirectory using virtual threads.
 * One virtual thread per node per query. Merges top-k via TopKMerger.
 */
public final class ScatterGatherExecutor {
    public ScatterGatherExecutor(NodeDirectory directory, Duration timeout);

    /**
     * Execute plan against all target nodes concurrently.
     * Partial results are returned if some nodes time out (with a WARNING log).
     */
    public SearchResult execute(List<LocalSearchRequest> plan, int k);
}

/** Resolves NodeId to a callable search endpoint. */
public interface NodeDirectory {
    NodeSearchClient clientFor(NodeId nodeId);
}

/** Thin call abstraction. In-process impl for tests; gRPC impl for production. */
public interface NodeSearchClient {
    SearchResult search(LocalSearchRequest request);
}
```

**Test class**: `ScatterGatherExecutorTest` (new, `@Tag("unit")`)

Uses `InProcessNodeDirectory` — a `NodeDirectory` backed by real `VectorCollection`s
running in the same JVM (no network). This is the primary integration vehicle.

Acceptance tests:
- `scatterGatherAcross3Nodes_returnsTopK` — 3 in-process nodes, 10K docs each,
  top-10 from scatter-gather matches brute-force top-10 on the merged dataset
- `partialTimeoutReturnsPartialResults` — one node's client sleeps past timeout;
  executor returns results from the two responding nodes
- `allNodesTimeOutReturnsEmpty`
- `singleNodeClusterWorksAsPassthrough` — degenerate 1-node case

**Benchmark class**: `ScatterGatherBenchmark` (`vectors-bench`)

| Benchmark | Nodes | N/node | D | k | Target |
|-----------|-------|--------|---|---|--------|
| `scatterGather_3nodes_FLAT` | 3 | 10K | 128 | 10 | < 5 ms p99 |
| `scatterGather_10nodes_FLAT` | 10 | 10K | 128 | 10 | < 10 ms p99 |
| `scatterGather_3nodes_HNSW` | 3 | 100K | 128 | 10 | < 20 ms p99 |

### 9.4 `GossipMembership` — centroid propagation

The `BuoyIndex` must be consistent across all nodes. Rather than a consensus protocol,
nodes gossip the current `BuoyIndex` version hash. On mismatch, nodes fetch the new
`BuoyIndex` from S3 (or the coordinator). The gossip is passive: no leader required.

```java
public interface ClusterMembership {
    Set<NodeId> liveNodes();
    void registerChangeListener(Consumer<MembershipEvent> listener);

    sealed interface MembershipEvent permits MembershipEvent.NodeJoined,
        MembershipEvent.NodeLeft, MembershipEvent.BuoyIndexUpdated {
        record NodeJoined(NodeId nodeId) implements MembershipEvent {}
        record NodeLeft(NodeId nodeId) implements MembershipEvent {}
        record BuoyIndexUpdated(String newVersionHash) implements MembershipEvent {}
    }
}

/** Static in-process membership for tests. No gossip protocol. */
public final class StaticClusterMembership implements ClusterMembership { ... }
```

### 9.5 `DistributedVectorCollection` — the public facade

```java
/**
 * Implements the full VectorCollection API across a distributed cluster.
 * Writes go to the local node's EventLog (S3 WAL). Reads scatter-gather
 * across cluster nodes via the BuoyIndex routing.
 *
 * Consistency model: read-your-own-writes within a session; eventual
 * consistency across nodes (followers lag by at most one WAL flush interval).
 */
public final class DistributedVectorCollection implements VectorCollection {
    public static Builder builder();

    // All VectorCollection methods implemented.
    // add() / delete() / upsert() → append to local S3 WAL → periodic local commit
    // search() → BuoyIndex.route() → ShardRouter.plan() → ScatterGather → merge
    // size() → sum of live node sizes (approximate, cached)
    // compact() → each node compacts its own shard
}
```

**Integration test class**: `DistributedVectorCollectionTest` (new, `@Tag("unit")`,
uses `InProcessNodeDirectory` + `HeapStorageBackend`)

Acceptance tests — Phase 3 gate:
- `addSearchConsistency_3nodes` — add 100 docs, `commit()`, search finds them
- `deletePropagatesToSearchResults` — delete 10 docs, commit, they don't appear in search
- `scatterGatherRecall_3nodes_1M_total` — 3 nodes × 333K docs, recall@10 ≥ 0.90
  with nprobe=32 (`@Tag("slow")`)
- `nodeFailureReturnsPartialResults` — one in-process node returns errors;
  result is non-empty and warning is logged
- `buoyIndexGossipConverges` — BuoyIndex update propagates to all nodes within 500 ms

---

## 10. Test-First Development Sequence

The following sequence is strict: no implementation code for step N+1 before all
tests for step N pass.

```
Step P1  VectorUtilBatchTest          (vectors-core)   must pass before K-means code
Step P2  KMeansTest                   (vectors-core)   must pass before CentroidIndex
Step P3  CentroidIndexTest            (vectors-core)   must pass before BuoyIndex
Step P4  StorageBackendContractTest   (vectors-storage) must pass before WAL
Step P5  WriteAheadLogTest            (vectors-storage) must pass before VectorEvent
Step P6  VectorEventCodecTest         (vectors-db)     must pass before SegmentTransfer
Step P7  SegmentTransferTest          (vectors-db)     must pass before IvfIndex
Step P8  BuoyIndexTest                (vectors-ivf)    must pass before IvfIndex
Step P9  IvfIndexIntegrationTest      (vectors-ivf)    PHASE 1 GATE
Step P10 TieredClusterTest            (vectors-ivf)    must pass before ScatterGather
Step P11 ScatterGatherExecutorTest    (vectors-distributed) PHASE 2/3 GATE
Step P12 DistributedVectorCollectionTest (vectors-distributed) PHASE 3 GATE
```

---

## 11. Phased Implementation Roadmap

### Phase 1 — IVF Foundation (vectors-core + vectors-storage + vectors-ivf)

**Goal**: Single-node IVF-flat search with SOAR spilling passes recall gate.
**Gate**: `IvfIndexIntegrationTest` all pass; `IvfIndexBenchmark.search_N1M_K1024_D128_nprobe32` < 5 ms p99.

| Component | Module | New files |
|-----------|--------|-----------|
| `batchDotProduct` / `batchSquaredL2` | vectors-core | VectorUtil additions |
| `KMeans` | vectors-core | `cluster/KMeans.java` |
| `CentroidIndex` | vectors-core | `cluster/CentroidIndex.java` |
| `StorageBackend` + backends | vectors-storage | `backend/*.java` |
| `SegmentedWriteAheadLog` | vectors-storage | `wal/*.java` |
| `BuoyIndex` | vectors-ivf | module scaffold + `BuoyIndex.java` |
| `ClusterPartition`, `IvfIndex` | vectors-ivf | `IvfIndex.java` |

### Phase 2 — Tiered Clusters + Event Sourcing (vectors-db + vectors-ivf)

**Goal**: Multi-pass cascade within clusters; VectorEvent / SegmentTransfer primitives in vectors-db.
**Gate**: `TieredClusterTest`; `SegmentTransferTest`; cascade recall@10 ≥ 0.98 at 4× compression vs float32.

| Component | Module | New files |
|-----------|--------|-----------|
| `VectorEvent` + `VectorEventCodec` | vectors-db | `event/*.java` |
| `SegmentExporter` / `SegmentImporter` | vectors-db | `transfer/*.java` |
| `TieredCluster` (T0 + T1 + T2) | vectors-ivf | `TieredCluster.java` |
| `IvfHnswIndex` | vectors-ivf | `IvfHnswIndex.java` |

### Phase 3 — In-Process Distributed Execution (vectors-distributed)

**Goal**: Multi-node in-process cluster with gossip membership; all `DistributedVectorCollectionTest` pass.
**Gate**: `ScatterGatherBenchmark.scatterGather_10nodes_FLAT` < 10 ms p99.

| Component | Module | New files |
|-----------|--------|-----------|
| `NodeId`, `ShardOwnership`, `ShardRouter` | vectors-distributed | module scaffold |
| `ScatterGatherExecutor` + `InProcessNodeDirectory` | vectors-distributed | `ScatterGatherExecutor.java` |
| `StaticClusterMembership` + `GossipMembership` | vectors-distributed | `membership/*.java` |
| `DistributedVectorCollection` | vectors-distributed | `DistributedVectorCollection.java` |
| `S3StorageBackend` (optional, compile-only AWS SDK) | vectors-distributed | `backend/S3StorageBackend.java` |

### Phase 4 — Network Transport (Future)

gRPC service definitions, TLS, node discovery via DNS-SRV or Kubernetes endpoints.
Not scheduled — deferred until Phase 3 is stable and a real deployment target exists.

---

## 12. Open Questions

1. **Retrain trigger**: When should the `BuoyIndex` be retrained? Options: (a) after N inserts,
   (b) when cluster imbalance ratio exceeds threshold, (c) manual. The SOAR paper uses a
   fixed offline build; DiskANN uses periodic incremental updates. Recommendation: manual
   retrain for Phase 1, automatic trigger in Phase 3.

2. **Write path**: Should `add()` / `upsert()` write to the S3 WAL synchronously (strong
   durability) or asynchronously (lower write latency)? Recommendation: synchronous WAL
   append (like Kafka `acks=1`) as the default; async as an opt-in.

3. **Cluster assignment for new vectors**: When a new vector is added, which cluster does it
   belong to? Options: (a) nearest centroid at insert time, (b) deferred to next retrain.
   Recommendation: (a) nearest centroid at insert time; cluster imbalance corrected at retrain.

4. **Replication factor**: Phase 3 designs for RF=1 (no replicas). RF>1 requires either
   synchronous dual-write or an async catch-up protocol. Deferred to Phase 4.

5. **Filter + IVF interaction**: Pre-filtering (restrict ANN search to matching ordinals per
   cluster) vs post-filtering (filter after ANN, existing behaviour). Pre-filtering within
   `ClusterPartition` enables better recall for selective filters but requires the filter
   executor to also be invoked at the cluster level.

6. **Quantizer-per-cluster vs global**: FAISS trains one PQ codebook per cluster for OPQ
   alignment. RaBitQ uses one global rotation. Recommendation: global RaBitQ for T0 (already
   implemented); per-cluster SQ8 for T1 (cluster distribution centred, better utilisation).
