# Distributed Vector Database: Design and Development Plan

> **Aspirational design, not released capability.** The 0.1.x release is
> single-process and publishes none of the distributed, server, cluster, or GPU
> modules. Scale, latency, cost, topology, and delivery-date figures in this
> document are design targets or estimates; they have not been validated by a
> reproducible release benchmark and must not be used in promotion.
>
**Status**: Design Draft (May 2026, revised after Turbopuffer ANN v3 deep-dive — see §16)
**Author**: java-vectors project
**Inputs**: `tiered-buoy-architecture.md`, `distributed-vector-search-analysis.md`, `design-strategy.md`, `competitive-analysis.md` §11 (enterprise-pattern adoption) and §12 (GPU roadmap), Turbopuffer `docs/architecture` and `blog/ann-v3` (May 2026)

> **Reading order**: §16 is the canonical Turbopuffer mapping and is **authoritative when it
> conflicts with earlier sections**. The interleaved revisions in §3.1, §6.1, §6.2, §8.5,
> §8.7, §8.8, and §12 already reflect §16; the rest of the doc is unchanged from the
> April 2026 draft.

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

The design has two interlocking halves:

1. **Algorithmic core** (§3, §5–§8): the three-layer IVF architecture — `BuoyIndex` (global
   routing), `SubBuoyTree` (hierarchical refinement), `TieredCluster` (per-cluster T0→T3
   cascade) — supported by primitive additions to `vectors-core`, `vectors-storage`,
   `vectors-db`.
2. **Cluster / deployment fabric** (§9): the enterprise-grade wrapper around the algorithmic
   core, adopting proven patterns from Apache Ignite 3, Hazelcast, Infinispan, EclipseStore
   + Eclipse Data Grid, and Chronicle Map. Covers lifecycle, membership, replication,
   state transfer, rolling upgrades, observability, and security. Mapping in §15.

The algorithmic core is unchanged from prior drafts. §9 (cluster fabric) and §15 (pattern
mapping) are the revisions driven by `competitive-analysis.md` §11.

### 1.1 Non-Goals

- No changes to the `VectorCollection` public API as seen by Spring AI / LangChain4j adapters
- **No JNI we maintain** — the runtime may optionally load `libcuvs.so` via the Panama-FFM
  bindings `cuvs-java` ships (see §12 and `vectors-gpu`), but no C/C++ source is built or
  maintained inside this repository
- `vectors-distributed` does **not** depend on `vectors-spring-ai` or `vectors-langchain4j`
- `vectors-distributed` does **not** depend on `vectors-gpu`; GPU acceleration is an
  orthogonal axis applied per-node, not a distribution concern

### 1.2 Revised position on consensus

Earlier drafts stated a flat "no consensus protocol" non-goal and named S3 compare-and-swap
as the sole coordination primitive. Adopting enterprise patterns from Apache Ignite 3
(§11 pattern #3) refines this:

- **Data durability** remains anchored on S3 / object-store PutObject (or any
  `NonBlockingStore` implementation). No synchronous multi-node write-quorum.
- **Cluster metadata** (BuoyIndex version, catalog, shard ownership, membership epoch) uses
  a **small Raft group** (JRaft embedded, single consensus group per cluster — the
  "Cluster Management Group" pattern). This prevents split-brain when two partitions attempt
  to rebalance or commit a new BuoyIndex concurrently.
- **Per-partition leadership**: within a partition, writes go to the primary owner and
  replicate asynchronously to backups with per-replica version counters (Hazelcast
  `PartitionReplicaVersionManager` pattern). No Raft per partition.

The result: Raft runs once per cluster over a small, infrequently-updated state (catalog +
topology); the vector data path remains free of consensus latency.

---

## 2. Design Principles

1. **Embedded first** — the single-node `VectorCollection` must work identically at every
   scale. The distributed layer is a thin shell, not a redesign. A `VectorClusterServer`
   instance runs inside the application JVM by default, auto-joins a cluster when peers are
   configured, and falls back to local-only mode when they are not (EclipseStore
   `EmbeddedStorage` + Ignite 3 `IgniteServer` pattern).
2. **Quantization = tier** — 1-bit RaBitQ codes live on every node (globally replicated);
   full-precision vectors live only where they're needed (SSD/S3).
3. **Object store as durable data floor** — data durability = `NonBlockingStore.put()` ack,
   backed by S3 / GCS / any object store. Cluster metadata is anchored by an embedded
   Raft group on top of the same store (see §1.2).
4. **Virtual threads for scatter-gather** — Java Loom eliminates thread-pool tuning.
5. **Pluggable SPIs, sensible defaults** — `DiscoveryStrategy` (Hazelcast pattern),
   `NonBlockingStore` (Infinispan pattern), `AuthorizationManager` (Infinispan pattern)
   are discovered via `ServiceLoader`. Core ships in-process and local-file implementations
   out of the box.
6. **Topology as a builder flag** — the collection builder accepts `CacheMode.LOCAL`,
   `CacheMode.REPLICATED`, or `CacheMode.DISTRIBUTED` (Infinispan pattern). Code that
   works against one mode works against all three.
7. **Deterministic startup, reverse shutdown** — components implement `VectorsComponent`
   with `startAsync` / `stopAsync`; the `LifecycleManager` brings them up in dependency
   order and tears them down in reverse (Ignite 3 pattern).
8. **Test-first** — each primitive has an acceptance test before the first implementation line.
9. **Benchmark-gated** — each phase has explicit JMH throughput/latency targets that must
   pass before the phase is considered complete.

---

## 3. Architecture Overview

The design composes two concentric shells:

- **Algorithmic shell** — three interlocking IVF layers (Global Buoy, Sub-Buoy Tree, Tiered
  Cluster) implemented in `vectors-ivf`. Described in §3.1 below.
- **Deployment shell** — lifecycle, membership, replication, state transfer, observability,
  and security — implemented in `vectors-distributed` as an enterprise-grade wrapper around
  the algorithmic core. Described in §3.4 and detailed in §9. Patterns adopted from
  Apache Ignite 3 / Hazelcast / Infinispan / EclipseStore + Eclipse Data Grid / Chronicle Map.

Both shells are buildable and testable independently. The algorithmic shell is a pure
library that runs fine single-node; the deployment shell wraps it into a cluster-capable
server.

### 3.1 The Three Algorithmic Layers

> **§16 alignment note**: Layers 1 and 2 are jointly the "ANN v3 hierarchical centroid
> tree" of Turbopuffer's design — a single tree with branching factor **100** between
> levels and **~100 vectors per leaf cluster**. Layer 1 (`BuoyIndex`) is the root of
> that tree; Layer 2 (`SubBuoyTree`) is the rest of it. They are separate types in the
> codebase only because Layer 1 has different residency rules (replicated to every node)
> and a SIMD-optimised single-step routing path. Conceptually they are one structure.

**Layer 1 — Global Buoy Index** (always-hot, JVM heap, **root of the centroid tree**)

A flat array of up to K root centroids that partitions the entire vector space into
Voronoi cells. Routes every query to the top-nprobe clusters via a single SIMD batch
dot-product. SOAR spilling handles Voronoi boundary cases with only 5–20% memory overhead.

K is chosen so the root level fits in L2 cache. For collections small enough that the
whole index fits in one level (N ≤ 10K with leaf=100), Layer 2 is a no-op and `BuoyIndex`
*is* the entire centroid tree. For larger N the tree extends downward through Layer 2
with the same branching factor of 100. Examples:

| N (vectors) | Tree depth | Leaves | Vectors/leaf | Root size (D=128, RaBitQ) |
|---|---|---|---|---|
| 10K | 2 (just BuoyIndex with K=100) | 100 | 100 | 1.6 KB |
| 1M | 3 (BuoyIndex K=100 + 1 Layer-2 level) | 10K | 100 | 1.6 KB |
| 100M | 4 (BuoyIndex + 2 Layer-2 levels) | 1M | 100 | 1.6 KB |
| 10B | 5 (BuoyIndex + 3 Layer-2 levels) | 100M | 100 | 1.6 KB |

> **The T0 global-replication advantage**: Because 1-bit RaBitQ codes (T0) are tiny —
> 16 MB for 1M 128-dim vectors, 1.6 GB for 1B — they are replicated to **every node**.
> The first-pass scoring that eliminates ≥90% of candidates is therefore always local.
> Only T1 rescore and T2 exact-score ever cross the network. This is why distributed
> search latency stays below 20 ms even at 100M+ vectors — the expensive computation
> never leaves the node that holds the data.

> **Compute-bound, not bandwidth-bound** (Turbopuffer ANN v3, May 2026): once T0 is
> 1-bit-quantised the inner loop reuses each fetched bit ~4× (RaBitQ §3.3.2 of the
> paper), giving ~64× higher arithmetic intensity than `f16` distance kernels. The
> bottleneck shifts from DRAM bandwidth to CPU. The JDK-25 Vector API
> `LongVector.bitCount()` lowers to AVX-512 `VPOPCNTDQ` (1 instruction/cycle pipelined,
> 3-cycle latency), which is what makes the JVM kernel competitive with native AVX-512.
> Verification protocol: a JMH microbenchmark and `-XX:+PrintAssembly` confirm the
> emitted code uses `VPOPCNTDQ`; degraded fallback (AVX2, no popcount) must be detected
> at startup and surfaced as a metric.

**Layer 2 — Hierarchical Sub-Cluster Tree** (JVM heap, per-cluster sub-buoys, branching=100)

Large clusters are recursively subdivided by `ClusterSplitter` using the Quake cost model:

```
C_i = A_i × λ(s_i)
```

where `A_i` is the cluster's measured access frequency and `λ(s_i)` is the profiled median
scan latency for its size. A cluster splits when `C_i > C_left + C_right + τ` (τ = 250 ns
default); merges when `C_i + C_j < C_merged + τ`. Splitting uses balanced k-means with
k=100; a three-phase estimate→verify→commit/rollback cycle prevents premature splits.

The resulting `SubBuoyTree` is a recursive tree of sub-buoys with **branching factor 100**
and a leaf-size target of **~100 vectors**. The branching factor matches the ratio between
adjacent memory tiers (DRAM≈10×–50× larger than L3, SSD≈10×–50× larger than DRAM), which
keeps each level resident in the cache it can afford to occupy:

```
Level 0 (root, BuoyIndex)         100 centroids   →  L1/L2 cache
Level 1                           10K centroids   →  L3 cache (~128 KiB at D=128 RaBitQ)
Level 2                           1M centroids    →  DRAM    (~16 MiB at D=128 RaBitQ)
Level 3 (leaves, ~100 vecs each)  100M leaves     →  SSD     (full-precision, paged)
```

The tree refines Layer 1 routing from root-cluster granularity down to leaf granularity
before dispatching to storage. Probe budget is **~500 leaves per query** to hit recall
target (Turbopuffer ANN v3 measured value), which sets the inner-loop bandwidth
requirement at ~6 MB/level when codes are 1-bit-quantised (vs. ~100 MB/level at f16).

**Layer 3 — Per-Cluster Tiered Storage** (the novel contribution)

Within each leaf cluster, vectors exist in four tiers by quantization fidelity:

| Tier | Name | Representation | Storage | Typical latency |
|------|------|----------------|---------|-----------------|
| T0 | Hot | 1-bit RaBitQ | JVM heap | < 1 µs |
| T1 | Warm | SQ8 (8-bit scalar) | Off-heap MemorySegment | ~ 1 µs |
| T2 | Cool | float32, mmap'd | Local NVMe SSD | 10–100 µs |
| T3 | Cold | float32, archived | Object storage (S3/GCS) | 50–200 ms |

Cross-tier links — **hyper doors** — are O(1) index arithmetic over the shared ordinal
space: `t1_address = sq8Segment.asSlice(ordinal * D)`. No hash lookup, no pointer chasing.

**Quantization IS the tier.** java-vectors' existing quantizers (RaBitQ, SQ8, float32)
provide exactly the three fidelity levels. The cascade already exists in the two-pass
search in `vectors-hnsw` and `vectors-vamana`; `TieredCluster` generalises it to four
tiers driven by access frequency.

The tier materialisation policy (`TierPolicy`) is access-frequency-driven:

```
T0: ALL clusters   (always materialised — tiny, 16 MB / 1M vecs)
T1: clusters with A_i > θ₁   (default 0.01)  — demand-loaded off-heap
T2: clusters with A_i > θ₂   (default 0.05)  — mmap'd from SSD
T3: ALL clusters   (always on S3 — source of truth)
```

Research shows ~15% of IVF clusters are accessed in a full day (CatapultDB, 2026). Only
that 15% needs T1 and T2 materialised; the other 85% are served from T0 for routing and
T3 for rare full-resolution queries.

### 3.2 Physical Layout

> The diagram below depicts the **legacy single-level** sizing (`K=1024 @ D=128`,
> ~524 KB) preserved for the existing `BuoyIndexBenchmark.route_K1024_D128_nprobe32`
> test fixture. Production deployments use the hierarchical centroid tree from §16.3
> with branching factor 100 between levels; root level shrinks to ~1.6 KB (RaBitQ-
> quantised) and additional levels live in `SubBuoyTree` (§8.5).

```
                         Query
                           |
                           v
                +-----------------------+
                |   Global Buoy Index   |  JVM Heap, ~524 KB @ K=1024 D=128
                |   K buoy vectors      |  < 1 µs routing, SIMD batch distance
                |   SOAR spill map      |  replicated to EVERY node
                |   cluster sizes/radii |
                +-----------+-----------+
                            |
                   route to top-nprobe clusters (Layer 1)
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
    +-------+-------+ +----+----+ +--------+--------+
    |  Cluster A    | | Cluster B| |  Cluster C      |
    |               | |         | |                  |
    |  Sub-buoys    | | Sub-buoys| | Sub-buoys       | <- SubBuoyTree (heap)
    |  (recursive)  | | (recurs.)| | (recursive)     |    refine to leaf
    |               | |         | |                  |
    | T0|T1|T2| T3  | | T0|T1|  | | T0|   | T3  |  | <- TieredCluster (Layer 3)
    | Hp|Oh|mm| S3  | | Hp|Oh|  | | Hp|   | S3  |  |    HyperDoor links tiers
    +-------+-------+ +----+----+ +--------+--------+
            |               |               |
            +-------+-------+-------+-------+
                    |               |
            +-------+-------+ +-----+-------+
            | Local SSD/mmap| | Object Store|   T2: mmap, T3: S3/GCS
            | (T2 clusters) | | (T3 always) |
            +---------------+ +-------------+
```

### 3.3 Distributed Query Path

```
Client query
  |
  v
BuoyIndex.route(query, nprobe, gamma)      -- Layer 1: < 1 µs SIMD, any node
  |  [top-nprobe cluster ids + SOAR spills]
  v
SubBuoyTree.refineToLeaves(clusterIds, q)  -- Layer 2: sub-cluster routing, heap
  |  [leaf sub-cluster ids, grouped by owner node]
  v
ScatterGatherExecutor (virtual threads)    -- one vthread per owning node
  |-- Node A: TieredCluster.search()      -- Layer 3: T0→T1→T2 cascade, < 2 ms
  |            T0 scoring always local (globally replicated 1-bit codes)
  |            T1/T2 local if cluster is hot; T3 fetch from S3 if cold
  |-- Node B: TieredCluster.search()
  |-- Node C: TieredCluster.search()
  |
  v
TopKMerger.merge(results, k)               -- heap merge, O(nodes × k × log k)
  |
  v
Client: top-k results with exact float32 scores
```

### 3.4 Deployment Shell (enterprise wrapper around the algorithmic core)

Every `vectors-distributed` node is a self-contained JVM process (or an embedded
participant in the application JVM) composed of the following components, each backed by a
proven enterprise pattern. §9 specifies each in detail; §15 maps them back to the
`competitive-analysis.md` §11 pattern inventory.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  VectorClusterServer  (embedded mode by default, or standalone process)      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  LifecycleManager  — DAG startup / reverse shutdown (Ignite 3)         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Discovery    │  │ ClusterSvc   │  │ CMG          │  │ Partition    │      │
│  │ (SPI:        │→ │ (SWIM gossip │→ │ (JRaft CMG   │→ │ Service      │      │
│  │  multicast / │  │  ScaleCube)  │  │  — metadata, │  │ (owned-      │      │
│  │  k8s / static│  │  liveness &  │  │  catalog,    │  │  partition   │      │
│  │ /consul)     │  │  failure det │  │  topology)   │  │  model,      │      │
│  │              │  │              │  │              │  │  Hazelcast)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  ALGORITHMIC CORE (§3.1)                                             │    │
│  │    BuoyIndex · SubBuoyTree · TieredCluster · HyperDoor               │    │
│  │    IvfIndex / IvfHnswIndex (per owned partition)                     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ State        │  │ NonBlocking  │  │ Catalog      │  │ Metrics &    │      │
│  │ Transfer     │  │ Store SPI    │  │ Manager      │  │ OTel         │      │
│  │ (segment     │  │ (local file /│  │ (rolling     │  │ (Micrometer  │      │
│  │  rebalance,  │  │  S3 / GCS /  │  │  upgrades,   │  │  + OTLP,     │      │
│  │  Infinispan) │  │  JDBC, Inf.) │  │  Ignite 3)   │  │  Infinispan) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Authorization (Subject + AuthorizationManager, Infinispan)          │    │
│  │  Transport (Netty direct-message codegen, Ignite 3 `@Transferable`)  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Deployment topologies supported** (all exposed via a `CacheMode` builder flag — Infinispan
pattern):

| Mode | Builder | Semantics | When |
|---|---|---|---|
| `CacheMode.LOCAL` | `VectorCollection.builder()...build()` | Single JVM, mmap directory. No cluster, no gossip. Zero new threads. | Default. Dev. Tests. Spring AI / LangChain4j single-instance apps. |
| `CacheMode.REPLICATED` | `VectorCollection.builder().mode(REPLICATED)...build()` | Full copy on every node. Gossip for membership, BuoyIndex gossip'd in full. No sharding. | Small collections (<10M vectors), read-heavy, HA-critical. |
| `CacheMode.DISTRIBUTED` | `VectorCollection.builder().mode(DISTRIBUTED)...ownersPerPartition(2).build()` | Partitions owned by N replicas; writes go to primary + async backups; shard rebalance on topology change. | The default for >10M vectors / write-heavy / elastic scale. |

---

## 4. Module Impact Summary

| Module | Change type | What changes |
|--------|-------------|--------------|
| `vectors-core` | Additions | `KMeans`, `CentroidIndex`, batch SIMD distance methods in `VectorUtil` |
| `vectors-storage` | Additions | `NonBlockingStore` SPI (§9.4 — Infinispan pattern), `HeapStore`, `LocalFileStore`, `SegmentedWriteAheadLog`. Retains the older `StorageBackend` interface as an adapter over `NonBlockingStore` for back-compat with existing `vectors-db`. |
| `vectors-db` | Additions | `VectorEvent` sealed hierarchy, `VectorEventCodec`, `SegmentExporter/Importer`; `IndexType.IVF_HNSW` enum value; `CacheMode` enum on the collection builder |
| `vectors-quantization` | No change | Consumed as-is by `TieredCluster` tiers |
| `vectors-hnsw` | No change | Used within clusters via `IvfHnswIndex` |
| `vectors-vamana` | No change | Used within clusters |
| `vectors-ivf` | **New module** | `BuoyIndex` (Layer 1), `SubBuoyTree` + `ClusterSplitter` (Layer 2), `HyperDoor`, `TieredCluster`, `TierPolicy`, `ClusterPartition`, `IvfIndex`, `IvfHnswIndex` (Layer 3) |
| `vectors-distributed` | **New module** | **Lifecycle**: `LifecycleManager`, `VectorsComponent`. **Membership**: `DiscoveryStrategy` SPI (multicast / static / k8s / consul), `SwimClusterService`, `ClusterManagementGroup` (JRaft). **Partition**: `PartitionService`, `OwnedPartition`, `PartitionReplicaVersionManager`. **Topology**: `CacheMode` honoured end-to-end, `StateTransferManager`. **Execution**: `ShardRouter`, `ScatterGatherExecutor`, `DistributedVectorCollection` (implements `VectorCollection`). **Persistence**: `S3Store` (`NonBlockingStore` impl), `SegmentedWriteAheadLog` adapter. **Operations**: `CatalogManager` (rolling upgrades), `MetricsRegistry` (Micrometer + OTLP), `AuthorizationManager`. **Transport**: `NettyTransport` + `@Transferable` direct-message codegen. |
| `vectors-gpu` | **New module (§12)** | Panama-FFM bindings to NVIDIA cuVS (CAGRA, IVF-PQ, IVF-FLAT, Brute Force). Feature-detected at runtime; transparent CPU fallback. Independent of `vectors-distributed`. |
| `vectors-bench` | Additions | `BuoyIndexBenchmark`, `IvfIndexBenchmark`, `ScatterGatherBenchmark`, `TieredClusterBenchmark`, `PartitionRebalanceBenchmark` |

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
     * Partial-object fetch. Returns the byte slice {@code [offset, offset+length)}
     * of the object at {@code key}. The implementation must use HTTP {@code Range:
     * bytes=offset-offset+length-1} for S3-class backends so no full object is
     * downloaded. Required by {@link HyperDoor} t3 reads and by the RaBitQ
     * confidence-interval rerank loop, which fetches small per-vector slices
     * scattered across large index objects.
     *
     * @throws IOException             on transport error
     * @throws IndexOutOfBoundsException if {@code offset < 0}, {@code length < 0},
     *         or {@code offset + length} exceeds the object size
     * @throws java.util.NoSuchElementException if {@code key} is absent
     */
    byte[] getRange(String key, long offset, int length) throws IOException;

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
- `HeapStorageBackend` — `ConcurrentHashMap<String,byte[]>`, CRC-less, for tests;
  `getRange` returns `Arrays.copyOfRange`
- `LocalFileStorageBackend` — wraps the existing file layout (file path = key);
  `getRange` uses `FileChannel.read(ByteBuffer, position)`
- `S3StorageBackend` — lives in `vectors-distributed` (Phase 3, requires AWS SDK);
  `getRange` issues a `GetObjectRequest` with `range("bytes=" + offset + "-" + (offset+length-1))`

**Test class**: `StorageBackendContractTest` (new, `@ParameterizedTest` over `HeapStorageBackend`
and `LocalFileStorageBackend`, `@Tag("unit")`)

Acceptance tests (same body for all in-JVM implementations):
- `putAndGetRoundTrip`
- `getMissingKeyReturnsNull`
- `listReturnsKeysUnderPrefix`
- `deleteRemovesKey`
- `conditionalPut_succeedsWhenKeyAbsent` — expectedEtag=null
- `conditionalPut_failsOnStaleEtag` — returns succeeded=false, data unchanged
- `conditionalPut_succeedsWithCurrentEtag` — sequential chain of CAS succeeds
- `getRange_partialFetchMatchesFullSlice` — random offsets/lengths against a 4 MB blob,
  every slice byte-equal to `Arrays.copyOfRange(get(key), offset, offset+length)`
- `getRange_zeroLengthReturnsEmpty`
- `getRange_pastEofThrows` — `offset+length > objectSize` → `IndexOutOfBoundsException`
- `getRange_missingKeyThrows` — `NoSuchElementException`

**Test class**: `S3StorageBackendTest` (new, `@Tag("integration")` — requires Docker)

Uses a static `LocalStackContainer` (Section 10 pattern). Runs the same contract tests
against `S3StorageBackend` pointing at LocalStack, plus:
- `getRange_usesHttpRangeHeader` — captures the actual HTTP request via LocalStack's
  request log and asserts the `Range: bytes=…` header is present (no full GET fallback)
- `getRange_concurrent_500requests` — 500 random ranges against a 64 MB blob; every
  response correct; total bytes transferred < 1 % of full-GET cost
- `concurrentPuts_noDataloss` — 10 threads × 100 PutObject, all keys readable after
- `conditionalPut_acrossContainerRestart` — simulates crash + replay; stale ETag rejected

### 6.2 `WriteAheadLog` — sequence-numbered durable journal with namespace prefix layout

Package: `com.integrallis.vectors.storage.wal`

**Per-namespace S3 prefix layout** (Turbopuffer-aligned; canonical reference is §16.2):

```
{root}/{namespace_id}/
    wal/
        000000000001.log        ← closed, indexed (■)
        000000000002.log        ← closed, indexed (■)
        000000000003.log        ← closed, NOT YET indexed (◈)
        000000000004.log        ← active, group-commit buffer; flushed every 1 s
    index/
        000000000001.idx        ← immutable run produced by the Indexer for wal/0001
        000000000002.idx
        ...
    manifest.json               ← which wal/* are ■ vs ◈; current index epoch
```

A query node scans the **unindexed tail** (◈ files) exhaustively for strong consistency,
and uses `index/*` for everything older (eventual consistency over the indexed prefix).
This is the same compute-storage split Turbopuffer documents in `docs/architecture`.

```java
public interface WriteAheadLog extends Closeable {
    /**
     * Append an entry. The call returns the assigned sequence number once the
     * entry is durable (its enclosing WAL object has been PutObject'd successfully
     * — at-most-once semantics for the caller). Concurrent {@code append} calls
     * within the configured group-commit interval are coalesced into a single
     * WAL object to amortise S3 PUT cost; see {@link #groupCommitInterval()}.
     */
    long append(byte[] entry) throws IOException;          // returns seq number
    Stream<WalEntry> readFrom(long fromSeqInclusive) throws IOException;
    long lastSequenceNumber();                             // -1 if empty

    /** The unindexed-tail seq numbers (◈). A query node must scan these for strong consistency. */
    long[] unindexedTailSeqs();

    /** Mark a closed WAL object as indexed (■). Called by the Indexer after publishing the run. */
    void markIndexed(long seqStartInclusive, long seqEndInclusive) throws IOException;

    void flush() throws IOException;
    Duration groupCommitInterval();                        // default 1 s

    record WalEntry(long sequenceNumber, byte[] data) {}
}
```

Implementation: `SegmentedWriteAheadLog` — entries are buffered in memory and the active
segment is flushed to the backend exactly once per **group-commit interval** (default 1 s,
matching Turbopuffer's per-namespace WAL cadence). Each closed segment is **capped at
512 MB** (Turbopuffer's documented per-write batch limit); writes that would overflow
trigger an early flush and start a new segment. Each entry has a 4-byte length prefix +
4-byte CRC32. A closed segment is immutable.

Constructor parameters:
- `groupCommitInterval` (default `Duration.ofSeconds(1)`) — sets the maximum write-side
  latency; smaller values give faster durability acks at higher PUT cost.
- `maxSegmentBytes` (default `512 * 1024 * 1024` = 512 MB)
- `maxEntriesPerSegment` (default `Integer.MAX_VALUE`)

Group-commit semantics: a single `Thread.sleep(groupCommitInterval)` worker drains the
in-memory queue, builds one WAL object, calls `StorageBackend.put`, and only then
releases the awaiting `append` callers. With one writer per namespace and 1 s cadence
this matches Turbopuffer's documented **10k writes/s/namespace ceiling**: each appended
entry is small (≤ a few KB), so 10k entries fit in one ~1–10 MB WAL object well below
the 512 MB cap.

**Test class**: `WriteAheadLogTest` (new, `@Tag("unit")`)

Acceptance tests:
- `appendThenReadRecoversEntry`
- `sequenceNumbersStrictlyMonotonic`
- `readFromZeroReturnsAll`
- `readFromMidSeqSkipsEarlierEntries`
- `segmentRolloverCreatesNewSegment` — write > 512 MB, verify two segment files appear
- `crcCorruptionDetectedOnRead` — flip a byte in a closed segment → IOException
- `emptyLogReturnsMinusOneSequence`
- `groupCommit_concurrent10kAppendsCoalesceIntoOneObject` — 10 000 vthreads all calling
  `append` within a 1 s window → exactly one PUT against the underlying `StorageBackend`
- `groupCommit_overflowingBatchTriggersEarlyFlush` — entries summing to > 512 MB within
  one interval → two PUTs, both within their 512 MB caps
- `unindexedTailSeqs_includesActiveAndUnmarked` — initial state lists all closed-but-
  unmarked segments + the active segment's seq range
- `markIndexed_movesSegmentOutOfTail` — after `markIndexed(1,1000)`, those seqs no
  longer appear in `unindexedTailSeqs`

**Benchmark class**: `WriteAheadLogBenchmark` (`vectors-bench`)

| Benchmark | Entry size | Backend | Target |
|-----------|-----------|---------|--------|
| `append_1KB_p50_1node_1ns` | 1 KB | LocalFile | p50 ≤ 5 ms (sub-1-s commit) |
| `append_500KB_p50_1node_1ns` | 500 KB | LocalFile | p50 ≤ 50 ms |
| `append_500KB_p50_1node_1ns_S3` | 500 KB | LocalStack S3 | p50 ≤ 285 ms (Turbopuffer parity) |
| `throughput_10k_appends_1ns` | 1 KB | LocalFile, vthreads | ≥ 10 000 appends/s/namespace |
| `throughput_10k_appends_1ns_S3` | 1 KB | LocalStack S3, vthreads | ≥ 10 000 appends/s/namespace |
| `scan_100K_entries` | 1 KB | LocalFile | ≥ 500 000 entries/s |

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

`vectors-ivf` implements all three buoy layers. It is the heart of the distributed design.

Dependencies: `vectors-core`, `vectors-storage`, `vectors-quantization`

| Layer | Types |
|-------|-------|
| Layer 1 — Global routing | `BuoyIndex`, `CentroidIndex` (from vectors-core) |
| Layer 2 — Hierarchical splitting | `ClusterSplitter`, `SubBuoyTree` |
| Layer 3 — Per-cluster tiered storage | `HyperDoor`, `TieredCluster`, `TierPolicy`, `ClusterPartition` |
| Index API | `IvfIndex` (flat), `IvfHnswIndex` (graph within cluster) |

### 8.2 `BuoyIndex` — Layer 1 Global Routing

Package: `com.integrallis.vectors.ivf`

```java
/**
 * The global centroid routing index. Always resident in JVM heap. Immutable once
 * built; rebuilt when cluster topology changes (splits/merges/retrain).
 *
 * Memory footprint at K=1024, D=128:
 *   buoys[]:         1024 * 128 * 4 =  512 KB
 *   spillTargets[]:  1024 * 4       =    4 KB
 *   clusterSizes[]:  1024 * 4       =    4 KB
 *   clusterRadii[]:  1024 * 4       =    4 KB
 *   Total:                          = ~524 KB  (fits in L2 cache)
 */
public final class BuoyIndex {
    /**
     * Train K buoys via k-means++ over a representative sample (n >= 256*K rows).
     * Uses real dataset vectors nearest to mathematical centroids (SPANN approach)
     * to avoid quantization artifacts.
     * When buildSoar=true, constructs the spill map via second-nearest-centroid
     * assignment for each training point (SOAR, ICML 2024).
     */
    public static BuoyIndex train(float[][] dataset, int k, SimilarityFunction metric,
                                   boolean buildSoar, long seed);

    /**
     * Route query to the nprobe nearest clusters, with SOAR boundary expansion.
     * Any cluster whose buoy distance is within (1+gamma)*nearestDist also has
     * its spillTarget added (deduplicated). gamma=0.0 disables SOAR expansion.
     */
    public int[] route(float[] query, int nprobe, float gamma);

    public int k();
    public SimilarityFunction metric();
    public float[][] buoyVectors();
    public int[] spillTargets();       // SOAR spill map: cluster -> spill cluster
    public int[] clusterSizes();       // for ClusterSplitter cost model
    public float[] clusterRadii();     // max distance to centroid, for boundary detection

    /** Encode to bytes for gossip propagation and persistence. */
    public byte[] encode();
    public static BuoyIndex decode(byte[] bytes);
}
```

**Test class**: `BuoyIndexTest` (`@Tag("unit")`)
- `trainProducesKBuoys` — `buoyVectors().length == k`
- `memoryFootprintWithinBound` — encoded size ≤ 524 KB at K=1024, D=128
- `routeReturnsNprobeClusterIds`
- `routeResultsSortedByAscendingDistance`
- `soarExpansionAddsBoundarySpillTarget` — synthetic boundary query retrieves true NN
  that plain nprobe routing misses
- `noSpillWhenFarFromBoundary` — `gamma=0.0` returns exactly nprobe ids
- `encodeDecodeRoundTrip` — all fields bit-identical after encode/decode
- `routeIsIdempotent`

**Benchmark class**: `BuoyIndexBenchmark` (`vectors-bench`, throughput mode)

| Benchmark | K | D | nprobe | Target |
|-----------|---|---|--------|--------|
| `route_K1024_D128_nprobe32` | 1024 | 128 | 32 | < 10 µs/call |
| `route_K1024_D128_nprobe64_soar` | 1024 | 128 | 64 | < 20 µs/call |
| `route_K16384_D768_nprobe64` | 16384 | 768 | 64 | < 200 µs/call |

### 8.3 `ClusterPartition` — posting list for one cluster

```java
/**
 * Immutable posting list for a single cluster. Ordinals are in the global
 * VectorCollection ordinal space, not cluster-local.
 */
public record ClusterPartition(
    int clusterId,
    float[] centroid,
    int[] ordinals,     // global ordinals into the parent VectorCollection
    int size            // == ordinals.length
) {
    public boolean isEmpty() { return size == 0; }
}
```

### 8.4 `ClusterSplitter` — Layer 2 Adaptive Splitting

Implements the Quake cost model to decide when to split or merge clusters. Splitting
produces sub-clusters whose `ClusterPartition`s become children in the `SubBuoyTree`.

```java
/**
 * Cost-model-driven cluster splitter based on Quake (OSDI 2025).
 *
 * Cost model:   C_i = A_i * lambda(s_i)
 *   A_i         = access frequency: fraction of recent queries hitting cluster i
 *   lambda(s_i) = measured median scan latency (ns) for cluster size s_i
 *
 * Split condition:  C_i > C_left + C_right + tau
 * Merge condition:  C_i + C_j < C_merged + tau
 *   tau = minimum improvement threshold in nanoseconds (default 250 ns)
 */
public final class ClusterSplitter {

    public record ClusterCost(
        int clusterId, double accessFrequency, double scanLatencyNs, double cost) {}

    /**
     * Estimate whether splitting partition is beneficial. Does not commit.
     * Uses balanced 2-means to propose left/right sub-clusters.
     */
    public SplitProposal propose(ClusterPartition partition, double tau);

    record SplitProposal(
        ClusterPartition left, ClusterPartition right,
        float[] subBuoy,          // centroid for the new sub-buoy node
        double estimatedImprovement,
        boolean beneficial        // true when improvement > tau
    ) {}

    /**
     * Three-phase split: estimate → verify with actual queries → commit or rollback.
     * Verify phase probes the candidate split with a sample of recent queries and
     * measures actual latency improvement. Rolls back if improvement < tau.
     * Returns the two new partitions on commit; singleton list with original on rollback.
     */
    public List<ClusterPartition> splitOrRollback(
        ClusterPartition partition, List<float[]> recentQueries, double tau);

    /**
     * Merge two clusters when C_i + C_j < C_merged + tau.
     * Returns the merged partition, or empty if merge is not beneficial.
     */
    public Optional<ClusterPartition> mergeIfBeneficial(
        ClusterPartition a, ClusterPartition b, double tau);

    /** Record a query hit for a cluster. Used to update the sliding-window A_i estimate. */
    public void recordAccess(int clusterId);

    /** Current access frequency estimate for a cluster (EMA over windowSize queries). */
    public double accessFrequency(int clusterId, int windowSize);

    /** Empirically profile scan latency for a given cluster size. */
    public long profileScanLatencyNs(int clusterSize, int warmupRuns, int measureRuns);
}
```

**Test class**: `ClusterSplitterTest` (`@Tag("unit")`)
- `splitReducesCostForLargeHighAccessCluster`
- `splitRollbackWhenActualImprovementBelowTau` — verify phase fails → original returned
- `mergeReducesCostForSmallColdClusters`
- `noSplitProposedWhenCostImprovementBelowTau` — small, cold cluster → `beneficial=false`
- `accessFrequencyUpdatesViaMovingAverage` — three accesses over window → frequency > 0

### 8.5 `SubBuoyTree` — Layer 2 Hierarchical Cluster Tree

```java
/**
 * Recursive tree of sub-buoys produced by ClusterSplitter. The root is the global
 * BuoyIndex (its centroids are the level-0 nodes of this tree). Each interior
 * node may be subdivided into up to BRANCHING_FACTOR children, forming a tree of
 * depth up to MAX_DEPTH (default 4). The tree IS the centroid hierarchy; see
 * §3.1 and §16 for the rationale.
 *
 * Default shape (Turbopuffer ANN v3 alignment):
 *   BRANCHING_FACTOR = 100  — matches the DRAM/SSD size ratio (10×–50×) so each
 *                              level naturally lives in the next memory tier
 *   LEAF_TARGET_VECS = 100  — leaf clusters hold ~100 vectors each
 *   MAX_DEPTH        = 4    — max 100^4 = 100M leaves → 10B vectors
 *
 * Memory at D=128, RaBitQ-quantised centroids (1 bit/dim → 16 B/centroid):
 *   level 0: 100        centroids → 1.6 KB    (L1 cache)
 *   level 1: 10 000     centroids → 160 KB    (L3 cache)
 *   level 2: 1 000 000  centroids → 16 MB     (DRAM)
 *   level 3: 100M       leaves    → SSD-resident only when materialised at T1
 *
 * Always resident in JVM heap up through the deepest level that fits. Thread-safe
 * for concurrent reads; writes lock the affected sub-tree only (not the root).
 *
 * Gossip note: the full tree is encoded as a structural skeleton (parent/child
 * pointers + centroid bytes) and gossip'd among nodes; for a 100M-leaf tree this
 * is ~16 MB. Only the diff (new/removed nodes) is gossip'd after each split/merge.
 */
public final class SubBuoyTree {
    public static final int   DEFAULT_BRANCHING_FACTOR = 100;
    public static final int   DEFAULT_LEAF_TARGET_VECS = 100;
    public static final int   DEFAULT_MAX_DEPTH        = 4;

    public BuoyIndex globalBuoyIndex();
    public int branchingFactor();
    public int leafCount();
    public int depth();              // current max depth

    /**
     * Refine an initial set of cluster ids (from BuoyIndex.route) to leaf
     * sub-cluster ids by descending the sub-buoy tree. Returns at most maxLeaves
     * leaf ids, still sorted by ascending distance to query.
     *
     * Default probe budget: ~500 leaves (Turbopuffer ANN v3 measured value
     * for 90–95 % recall@10 across {@code D=1024} workloads; smaller D can
     * use proportionally fewer probes).
     */
    public int[] refineToLeaves(int[] clusterIds, float[] query, int maxLeaves);

    /** Register a k-way split. Adds {@code children} as children of clusterId; removes clusterId leaf. */
    public void split(int clusterId, List<ClusterPartition> children, float[][] subBuoys);

    /** Register a merge. Removes the listed clusterIds; adds {@code merged} as a new leaf. */
    public void merge(int[] clusterIds, ClusterPartition merged);

    /** Encode the full tree (global buoy index + all sub-buoy nodes) for gossip/persistence. */
    public byte[] encode();
    public static SubBuoyTree decode(byte[] bytes);
}
```

**Test class**: `SubBuoyTreeTest` (`@Tag("unit")`)
- `flatTreeRoutesIdenticallyToBuoyIndex` — no splits, `refineToLeaves` == `BuoyIndex.route`
- `splitThenRefine_routesToCorrectSubCluster` — split with k=100 children (not 2)
- `mergeThenRefine_treatedAsSingleLeaf`
- `depthLimitEnforced` — split attempt beyond MAX_DEPTH throws `IllegalStateException`
- `encodeDecodeRoundTrip` — all sub-buoy nodes and global index bit-identical after decode
- `concurrentReadsDuringWrite` — reads return consistent snapshot; no ConcurrentModificationException
- `levelMemoryBudget_atDepth4_undersizedForL3` — at D=128 RaBitQ, levels 0–1 fit ≤ 200 KB
  (must remain L3-resident); level 2 ≤ 20 MB (DRAM); the test hard-fails if a future
  tweak inflates these past their tier budgets

### 8.6 `HyperDoor` — cross-tier link record

```java
/**
 * Cross-tier link for a single cluster ordinal. All tiers share the same ordinal
 * space. Physical tier offsets are computed by O(1) arithmetic — no hash lookup,
 * no pointer chasing:
 *   T0: bitSet.get(t0BitOffset)
 *   T1: sq8Segment.asSlice(t1ByteOffset, D)
 *   T2: vectorsMmap.asSlice(t2FileOffset, D * Float.BYTES)
 *   T3: s3Backend.get(objectKey, t3ObjectOffset, D * Float.BYTES)
 *
 * In practice, since ordinals are contiguous within each tier, the HyperDoor
 * degenerates to stride arithmetic: t1ByteOffset = ordinal * D, etc.
 * The record is kept explicit for documentation and for non-contiguous layouts
 * (after compaction gaps or partial tier materialisation).
 *
 * Fields with value -1 indicate the tier is not locally materialised.
 * T0 (t0BitOffset) and T3 (t3ObjectOffset) are always >= 0.
 */
public record HyperDoor(
    int  clusterOrdinal,     // position within this cluster's ordinal space
    long t0BitOffset,        // bit offset in the 1-bit RaBitQ bitset (always present)
    int  t1ByteOffset,       // byte offset in the SQ8 MemorySegment (-1 = not materialised)
    long t2FileOffset,       // byte offset in the mmap'd vectors.bin (-1 = not loaded)
    long t3ObjectOffset      // byte offset in the S3 object key (always present)
) {
    public boolean hasT1() { return t1ByteOffset  >= 0; }
    public boolean hasT2() { return t2FileOffset  >= 0; }
}
```

**Test class**: `HyperDoorTest` (`@Tag("unit")`)
- `strideArithmetic_T0T1T2T3_offsetsConsistentWithOrdinal`
- `missingT1ReportedCorrectly` — `t1ByteOffset=-1` → `hasT1()==false`
- `missingT2ReportedCorrectly`

### 8.7 `TieredCluster` — Layer 3 per-cluster four-tier cascade

```java
/**
 * Per-cluster tiered storage with four quantization tiers linked by HyperDoor
 * ordinal arithmetic. The search cascade is driven by RaBitQ confidence-interval
 * rerank (Turbopuffer ANN v3, May 2026; canonical reference §16.3):
 *
 *   T0  : score every ordinal with the RaBitQ 1-bit kernel; for each ordinal i
 *         this returns NOT a scalar but an interval [lo_i, hi_i] bounding the
 *         true distance d_i. Costs ~D/8 bytes of bandwidth per ordinal.
 *   Cut : maintain a running top-k threshold τ_k = the k-th smallest hi seen
 *         so far. Any ordinal whose lo_i > τ_k is dropped: it cannot belong
 *         to the true top-k.
 *   T1/T2/T3 rerank: ONLY ordinals whose interval [lo_i, hi_i] straddles τ_k
 *         are reranked at higher precision. Empirically on Turbopuffer's
 *         workloads this is < 1 % of the original candidates.
 *
 *   T0 → (T1 if materialised) → (T2 if materialised) → (T3 via getRange)
 *
 * Inner-loop note (compute-bound; see §3.1): the T0 kernel is implemented with
 * jdk.incubator.vector.LongVector and LongVector.bitCount(), which on AVX-512
 * lowers to VPOPCNTDQ (1 ins/cycle pipelined). On AVX2 hardware it falls back
 * to a 4-bit-table popcount and the throughput drops ~2x. The fallback is
 * detected at startup and surfaced via the {@code vectors.simd.popcount}
 * metric so deployments can size accordingly.
 *
 * Data-read example (50K-vector cluster, D=128, k=10, rerank-ratio = 0.01):
 *   Brute-force float32:           50K * 128 * 4 =  25.6 MB
 *   T0 (RaBitQ) intervals:         50K * 128 / 8 =   800 KB  → ~500 candidates straddle τ_k
 *   T1 (SQ8) rerank, 1 % of T0:    500 * 128     =    64 KB  → ~50 still ambiguous
 *   T2 (float32) rerank, top-50:   50  * 128 * 4 =    25 KB  → return top-10
 *   Total read:                                  =   ~890 KB  (29x reduction)
 *
 * The "1 %" rerank ratio is the contract with TieredClusterTest; if the kernel
 * becomes loose-bounded enough that more candidates straddle τ_k, the workload
 * regresses toward the bandwidth-bound regime and the budget is blown.
 */
public final class TieredCluster {
    public int clusterId();
    public int size();
    public boolean hasT1();          // SQ8 codes loaded in off-heap MemorySegment
    public boolean hasT2();          // float32 vectors mmap'd from SSD

    /**
     * Confidence-interval cascade search. Tier selection is automatic based on
     * materialisation. Falls back gracefully when higher tiers are not present:
     * if T1 is absent the rerank goes straight to T2; if T2 is absent it goes
     * straight to T3 (range-GET). The {@code rerankRatioCap} parameter gates
     * how many candidates are eligible for rerank — exceeding it forces a
     * widening of τ_k (recall-preserving) rather than blowing the bandwidth
     * budget.
     */
    public SearchResult search(float[] query, int k, CascadeParams params);

    /** Demand-materialise T1 (SQ8 codes) into off-heap MemorySegment. */
    public void materializeT1(SQ8ClusterSource source);

    /** Memory-map T2 (float32) from a local cluster vectors.bin file. */
    public void materializeT2(Path clusterVectorsFile);

    /**
     * Fetch a per-ordinal slice from T3 (S3) on demand using
     * {@link StorageBackend#getRange}. This is the inner loop of cold reranks:
     * for each rerank-eligible ordinal compute its byte offset
     * (HyperDoor.t3ObjectOffset + ordinal * D * Float.BYTES) and issue
     * one range-GET. Concurrent vthreads issue these in parallel.
     */
    public float[] fetchOrdinalFromT3(StorageBackend s3, String objectKey, int ordinal)
        throws IOException;

    /** Bulk-fetch T2 from T3 (used at materialisation time, not per-query). */
    public void fetchFromT3(StorageBackend s3, String objectKey, Path localPath)
        throws IOException;

    /** Evict T1 off-heap segment. T0 and T2 are unaffected. */
    public void evictT1();

    /** Unmap T2 mmap'd segment. T0, T1, and T3 are unaffected. */
    public void evictT2();

    /** Record a query hit; used by TierPolicy and ClusterSplitter. */
    public void recordAccess();

    /** Exponential moving average access frequency over the last windowSize queries. */
    public double accessFrequency(int windowSize);

    /** Median scan latency sampled over recent search() calls (nanoseconds). */
    public long medianScanLatencyNs();

    record CascadeParams(
        int   t0Candidates,    // initial T0 candidate cap (default: max(1000, k*100))
        float rerankRatioCap,  // hard ceiling on (rerank candidates / t0Candidates),
                               // default 0.01 → < 1 % rerank ratio
        float gamma            // SOAR boundary tolerance passed through from routing
    ) {}
}
```

**Test class**: `TieredClusterTest` (`@Tag("unit")`) — in-JVM, no Docker

- `t0OnlySearch_returnsApproxResults` — only T0 materialised, recall@10 ≥ 0.70
- `t0T2Search_returnsHighRecall` — T0 + T2, recall@10 ≥ 0.95 (Phase 1 gate)
- `t0T1T2Cascade_returnsHighRecall` — full cascade, recall@10 ≥ 0.98 (Phase 2 gate)
- `confidenceIntervalCascade_rerankRatioBelowOnePercent` — across 1 000 random queries
  on a 50K-vector cluster, every query reranks ≤ 0.01 × t0Candidates ordinals
- `confidenceIntervalCascade_widensTauWhenRatioExceeded` — synthetic cluster with very
  loose [lo, hi] bounds; the algorithm must widen τ_k rather than rerank > 1 %
- `cascadeDataReadBelow1MB_50KCluster` — measure bytes read ≤ 900 KB for 50K cluster
- `fetchOrdinalFromT3_singleRange` — **mocked** `StorageBackend.getRange` returns the
  exact slice; arithmetic for a chosen ordinal verified byte-for-byte
- `fetchFromT3ThenSearch` — **mocked** `StorageBackend` (no Docker), T2 fetched and mmap'd
- `evictT1ThenSearch_fallsBackToT0T2` — after eviction, search still returns results
- `accessFrequencyUpdatesCorrectly` — 10 accesses, window=100 → frequency ≈ 0.10
- `hyperDoorOffsetArithmetic_correctForAllTiers`
- `t0Kernel_usesVectorPopcount_orFailsLoudly` — `assumeTrue` on AVX-512 capability;
  `-XX:+PrintAssembly` snapshot asserted to contain `VPOPCNTDQ` (CI annotation)

**Test class**: `TieredClusterFetchT3Test` (`@Tag("integration")`) — requires Docker

Uses a static `LocalStackContainer`. Validates `fetchFromT3` against a **real S3 API**
(not a mock), covering multipart objects, network interruption retry, and mmap correctness
after download:
- `fetchFromT3_singlePartObject_searchSucceeds` — cluster < 5 MB, single PutObject
- `fetchFromT3_largeCluster_multipartObject_searchSucceeds` — cluster > 8 MB, multipart
- `fetchFromT3_writesLocalFileBeforeMmap` — verifies file exists at `localPath` after fetch
- `fetchFromT3_retryOnTransientS3Error` — LocalStack container paused briefly mid-download,
  retry succeeds and result is correct
- `fetchFromT3ThenEvictT2_coldRefetch` — evict T2, re-fetch from T3, search still correct

**Benchmark class**: `TieredClusterBenchmark` (`vectors-bench`)

| Benchmark | N/cluster | Cascade | Target |
|-----------|-----------|---------|--------|
| `t0Only_N50K_D128` | 50K | T0 | < 500 µs |
| `t0T2_N50K_D128` | 50K | T0→T2 | < 2 ms p99 |
| `t0T1T2_N50K_D128` | 50K | T0→T1→T2 | < 2 ms p99 |
| `dataReadBytes_N50K_vs_bruteForce` | 50K | T0→T1→T2 | ≤ 1 MB (vs 25.6 MB BF) |

### 8.8 `TierPolicy` — access-frequency-driven materialisation

```java
/**
 * Drives automatic T1/T2 materialisation and eviction based on cluster
 * access frequency measured by TieredCluster.accessFrequency(), with an
 * explicit operator-controlled pin override (Turbopuffer "pinning" pattern;
 * see §16.5).
 *
 * Formal model (FaTRQ, 2026):
 *   T1 materialised when: A_i > theta_1   (default 0.01)
 *   T2 materialised when: A_i > theta_2   (default 0.05)
 *   T0 and T3 always present.
 *
 * Pin override: pin(clusterId, tier) forces the cluster to remain materialised
 * at {@code tier} or higher, bypassing the θ thresholds and ineligible for
 * eviction. Used by latency-sensitive tenants who do not want EMA-driven
 * cold-starts.
 *
 * Working set insight: ~15% of clusters are accessed in a full day (CatapultDB).
 * Configuring theta_2 = 0.05 materialises T2 for the hottest ~15% of clusters
 * and keeps T1 for the next ~10%, total warm set = ~25%.
 *
 * Pinning cap (per Turbopuffer parity): no more than 256 pinned clusters per
 * collection; exceeding this throws {@link IllegalStateException} from
 * {@link #pin}. The cap is configurable via the Builder.
 */
public final class TierPolicy {
    public static final int DEFAULT_MAX_PINS = 256;

    public static TierPolicy auto();   // theta_1=0.01, theta_2=0.05, max 256 pins

    public enum Decision { MATERIALISE, EVICT, NO_CHANGE }
    public enum Tier { T0, T1, T2, T3 }

    public Decision evaluateT1(TieredCluster cluster, int windowSize);
    public Decision evaluateT2(TieredCluster cluster, int windowSize);

    /**
     * Pin a cluster at {@code tier} (or higher). The cluster will be force-
     * materialised on the next applyAll cycle and protected from eviction
     * until {@link #unpin} is called. Idempotent: pinning at a higher tier
     * upgrades the existing pin; pinning at a lower tier is a no-op.
     *
     * @throws IllegalStateException if the per-collection pin cap is exceeded
     */
    public void pin(int clusterId, Tier tier);

    /** Remove the pin for a cluster. Subsequent applyAll calls re-evaluate via θ. */
    public void unpin(int clusterId);

    /** Snapshot of currently pinned cluster ids and their pinned tier. */
    public Map<Integer, Tier> pins();

    /** Apply policy to all clusters in a collection. Call periodically (e.g., every 60 s). */
    public void applyAll(Collection<TieredCluster> clusters, int windowSize,
                         StorageBackend s3, Path localSsdRoot) throws IOException;

    public static Builder builder();
    public static final class Builder {
        public Builder t1Threshold(double theta1);
        public Builder t2Threshold(double theta2);
        public Builder windowSize(int queries);
        public Builder maxPins(int maxPins);     // default 256
        public TierPolicy build();
    }
}
```

**Test class**: `TierPolicyTest` (`@Tag("unit")`)
- `hotCluster_materialisesBothT1AndT2` — A_i=0.10 > theta_2 → both materialise
- `warmCluster_materialisesT1Only` — A_i=0.03 > theta_1 but < theta_2 → T1 only
- `coldCluster_evictsBoth` — A_i=0.001 < theta_1 → evict T1 and T2
- `applyAll_materialisesCorrectFraction` — 100 clusters, 15 hot → 15 with T2
- `pinnedCluster_remainsMaterialised_evenWhenCold` — pin(c, T2) then drop A_c to 0;
  applyAll must NOT evict T1/T2 for `c`
- `pinUpgrade_isIdempotent` — pin(c, T1) then pin(c, T2) → effective tier = T2; pin
  count remains 1
- `unpin_restoresEMABehaviour` — after unpin(c) and one applyAll cycle with A_c=0,
  T1 and T2 are evicted as expected
- `pinCap_enforced_throwsAt257thPin` — `IllegalStateException` from the 257th pin
  with default `maxPins=256`

### 8.9 `IvfIndex` — flat IVF (Phase 1 search target)

```java
public final class IvfIndex implements Closeable {
    public static IvfIndex build(VectorCollection collection, IvfBuildParams params);
    public SearchResult search(IvfSearchRequest request);
    public BuoyIndex buoyIndex();
    public SubBuoyTree subBuoyTree();
    public ClusterPartition partition(int clusterId);
    public int clusterCount();
}

public record IvfBuildParams(
    int k, int maxIter, float gamma, boolean buildSoar, long seed) {}

public record IvfSearchRequest(
    float[] query, int k, int nprobe, float gamma, float minScore) {}
```

**Integration test class**: `IvfIndexIntegrationTest` (`@Tag("unit")`)
- `buildAndSearchFlat_10K_K64` — recall@10 ≥ 0.90, nprobe=8
- `buildAndSearchFlat_100K_K316_nprobe32` — recall@10 ≥ 0.95 (`@Tag("slow")`)
- `nprobeEqualsKGivesBruteForceRecall` — recall@10 ≥ 0.999
- `soarExpansionImprovesRecallAtBoundary` — gamma=0.2 recall ≥ gamma=0.0 recall

**Benchmark class**: `IvfIndexBenchmark` (`vectors-bench`, `@Tag("slow")`)

| Benchmark | N | K | D | nprobe | Target |
|-----------|---|---|---|--------|--------|
| `search_N1M_K1024_D128_nprobe32` | 1M | 1024 | 128 | 32 | < 5 ms p99 |
| `search_N1M_K1024_D128_nprobe64` | 1M | 1024 | 128 | 64 | < 8 ms p99 |
| `search_N100M_K10000_D128_nprobe64` | 100M | 10000 | 128 | 64 | < 20 ms p99 |

### 8.10 `IvfHnswIndex` — graph within clusters (Phase 2)

Same routing (BuoyIndex + SubBuoyTree) as `IvfIndex`, but replaces the brute-force scan
within each cluster with an HNSW graph. The tiered cascade in `TieredCluster` drives graph
traversal using T0 scores for neighbor selection, T1 for beam rescoring, T2 for final exact
scores. Adds `IndexType.IVF_HNSW` to the `IndexType` enum in `vectors-db`.

```java
// In vectors-db, vectors-db's IndexType enum gets:
// IVF_HNSW   -- IVF routing + per-cluster HNSW + four-tier TieredCluster

// User-facing builder API (from tiered-buoy-architecture.md §8.3):
VectorCollection collection = VectorCollection.builder()
    .dimension(128)
    .metric(SimilarityFunction.COSINE)
    .indexType(IndexType.IVF_HNSW)
    .quantizer(QuantizerKind.RABITQ)               // T0: 1-bit first pass
    .hnswParams(HnswParams.builder()
        .m(16).efConstruction(100).build())         // per-cluster HNSW
    .ivfParams(IvfParams.builder()
        .buoyCount(1024)
        .nprobe(32)
        .gamma(0.2f)                               // SOAR boundary tolerance
        .build())
    .tierPolicy(TierPolicy.builder()
        .t1Threshold(0.01).t2Threshold(0.05)
        .windowSize(10_000).build())
    .build();
```

---


## 9. New Module — `vectors-distributed`

### 9.1 Purpose and Boundaries

`vectors-distributed` is the cluster fabric around the algorithmic core (`vectors-ivf`).
It alone knows about nodes, network addresses, cluster membership, partitions, replication,
state transfer, rolling upgrades, observability, and security. Every component in the
module is designed to be:

- **Optional at single-node**: `CacheMode.LOCAL` collections never start the cluster stack.
- **Embeddable**: `VectorClusterServer.start(name, config, workDir)` runs inside the
  application JVM without a separate process (EclipseStore + Eclipse Data Grid / Ignite 3
  `IgniteServer` pattern).
- **In-process testable**: every SPI has an in-JVM implementation so unit tests never
  require network or Docker. Integration tests tagged `@Tag("integration")` exercise real
  object storage via LocalStack and real Netty transport on loopback.

**Dependencies** (`vectors-distributed/build.gradle.kts`):

- `vectors-core`, `vectors-storage`, `vectors-db`, `vectors-ivf` (internal)
- `io.scalecube:scalecube-cluster` (SWIM gossip — §9.5)
- `com.alipay.sofa:jraft-core` (embedded Raft for CMG — §9.6)
- `io.netty:netty-all` (transport — §9.13)
- `io.micrometer:micrometer-core` + `micrometer-registry-otlp` (metrics — §9.11)
- Optional (compile-only): `software.amazon.awssdk:s3` (S3 `NonBlockingStore` — §9.9)

### 9.2 Core Identifiers and Value Types

Package: `com.integrallis.vectors.distributed`

```java
/** Opaque cluster-unique node identifier. Immutable, gossipable. */
public record NodeId(String id, InetSocketAddress address) {}

/** Partition id within a collection. Partitions are the unit of ownership and rebalance. */
public record PartitionId(String collection, int index) {}

/** A partition's ownership state: one primary and zero-or-more backups. */
public record PartitionOwners(NodeId primary, List<NodeId> backups) {
    public List<NodeId> all() {
        var out = new ArrayList<NodeId>(backups.size() + 1);
        out.add(primary); out.addAll(backups); return out;
    }
}

/** Monotonically-increasing cluster state epoch, incremented on any topology change. */
public record ClusterEpoch(long value) implements Comparable<ClusterEpoch> {
    public ClusterEpoch next() { return new ClusterEpoch(value + 1); }
    @Override public int compareTo(ClusterEpoch o) { return Long.compare(value, o.value); }
}

/** Per-replica version counter for anti-entropy (Hazelcast PartitionReplicaVersionManager). */
public record ReplicaVersion(PartitionId partition, int replicaIndex, long version) {}
```

### 9.3 Lifecycle — `VectorsComponent` + `LifecycleManager`

**Adopts Ignite 3 pattern #2** (`competitive-analysis.md` §11): components declare their
start/stop contract, the `LifecycleManager` orchestrates a DAG startup and reverse shutdown.

```java
/** Every cluster component implements this contract. */
public interface VectorsComponent {
    String name();
    List<String> dependsOn();   // names of components that must start first

    CompletionStage<Void> startAsync(ComponentContext ctx);
    CompletionStage<Void> stopAsync();     // invoked in reverse dependency order
    default void beforeNodeStop() {}       // synchronous hook for draining work
}

public final class LifecycleManager {
    public LifecycleManager register(VectorsComponent component);
    public CompletionStage<Void> startAll(Duration timeout);
    public CompletionStage<Void> stopAll(Duration timeout);
    public ComponentState state(String componentName);
}
```

**Test class**: `LifecycleManagerTest` (`@Tag("unit")`)
- `startAll_respectsDependencyOrder`
- `stopAll_reverseOrderOfStart`
- `startFailure_stopsAlreadyStartedComponentsThenFails`
- `circularDependency_rejectedAtRegistration`

### 9.4 Discovery SPI — `DiscoveryStrategy`

**Adopts Hazelcast pattern #5** (`competitive-analysis.md` §11). Discovery is pluggable
via `ServiceLoader`. Core ships multicast and static implementations; external impls for
Kubernetes, Consul, etc. live in separate optional modules.

```java
public interface DiscoveryStrategy {
    String name();    // e.g. "multicast", "static", "kubernetes", "consul"

    /** Seeds for initial cluster join. Called by SwimClusterService during startup. */
    CompletionStage<List<InetSocketAddress>> discoverSeeds();

    /** Register this node's address with the discovery backend (if applicable). */
    CompletionStage<Void> registerSelf(NodeId self);

    /** Deregister on graceful shutdown. */
    CompletionStage<Void> deregisterSelf(NodeId self);
}

public final class MulticastDiscoveryStrategy implements DiscoveryStrategy { ... }
public final class StaticDiscoveryStrategy implements DiscoveryStrategy {
    public StaticDiscoveryStrategy(List<InetSocketAddress> seeds) { ... }
}
```

**Test class**: `DiscoveryStrategyContractTest` (`@ParameterizedTest`, `@Tag("unit")`)
- `discoverSeeds_returnsConfiguredSeedsForStatic`
- `registerThenDeregister_roundTrip`

### 9.5 Cluster Service — `SwimClusterService` (ScaleCube)

**Adopts Ignite 3 pattern #3a** (SWIM gossip for liveness). Wraps `io.scalecube.cluster`.

```java
public interface ClusterService extends VectorsComponent {
    Set<NodeId> liveMembers();
    NodeId localNode();
    void addMembershipListener(MembershipListener listener);

    interface MembershipListener {
        void onJoined(NodeId node);
        void onLeft(NodeId node);
        void onSuspect(NodeId node);
    }
}

public final class SwimClusterService implements ClusterService {
    public SwimClusterService(DiscoveryStrategy discovery, NodeId self,
                               Duration failureDetectInterval);
}
```

**Test class**: `SwimClusterServiceTest` (`@Tag("unit")`)
- `threeNodes_allSeeEachOther` (in-JVM via loopback)
- `nodeCrash_othersObserveLeftWithinFiveIntervals`
- `networkPartition_bothSidesSuspectOtherWithinTimeout`

### 9.6 Cluster Management Group — `ClusterManagementGroup` (JRaft)

**Adopts Ignite 3 pattern #3b** (Raft for cluster metadata). A single Raft consensus group
per cluster (name **CMG** — Cluster Management Group) holds the authoritative state for:

- cluster topology and `ClusterEpoch`,
- the `PartitionOwners` assignment for every partition,
- the catalog (collections, schemas, rolling-upgrade versions — §9.10),
- persistent cluster-level config overrides.

Three voting members by default; the remaining cluster nodes are learners. This bounds
Raft traffic regardless of cluster size. Data-path writes never touch CMG; only topology
and schema changes do.

```java
public interface ClusterManagementGroup extends VectorsComponent {
    /** Read the current cluster state (monotonic). */
    CompletionStage<ClusterState> readState();

    /** Submit a metadata mutation; returns when the entry has been committed by majority. */
    <T> CompletionStage<T> submit(MetadataCommand<T> command);

    boolean isLeader();
    Optional<NodeId> currentLeader();
}

public sealed interface MetadataCommand<T> permits
    MetadataCommand.AddCollection, MetadataCommand.RemoveCollection,
    MetadataCommand.UpdatePartitionOwners, MetadataCommand.BumpCatalogVersion {
    record AddCollection(String name, CollectionDescriptor desc)
        implements MetadataCommand<Boolean> {}
    record UpdatePartitionOwners(PartitionId p, PartitionOwners owners, ClusterEpoch epoch)
        implements MetadataCommand<Boolean> {}
    /* ... */
}
```

**Test class**: `ClusterManagementGroupTest` (`@Tag("unit")`)
- `threeNodeMajority_commitSucceeds` (in-JVM JRaft)
- `oneNodeDown_majorityStillCommits`
- `twoNodesDown_submitFailsAfterTimeout`
- `leaderElectionAfterCrash`

### 9.7 Partition Service — owned partitions, versioned replicas

**Adopts Hazelcast pattern #4** (`competitive-analysis.md` §11): consistent-hash assignment
of partitions to owners; writes go to the primary and fan out async to backups; per-replica
version counters enable anti-entropy without Merkle trees.

```java
public interface PartitionService extends VectorsComponent {
    int partitionCount(String collection);
    PartitionOwners owners(PartitionId partition);
    boolean isLocalPrimary(PartitionId partition);
    boolean isLocalBackup(PartitionId partition);

    /** Writes go here from the data path. Fans out to backups asynchronously. */
    <R> CompletionStage<R> runOnPrimary(PartitionId p, PartitionOperation<R> op);

    /** Anti-entropy: query a peer for version vectors; ship deltas. */
    CompletionStage<Void> reconcile(PartitionId p);
}

public interface PartitionReplicaVersionManager {
    ReplicaVersion current(PartitionId p, int replicaIndex);
    void bumpVersion(PartitionId p, int replicaIndex);
    ReplicaVersionVector vectorFor(NodeId node);
}
```

The partition-to-owner mapping is computed from the CMG-held `PartitionOwners` assignment;
rebalance is performed by `StateTransferManager` (§9.8). Partitions within a collection map
1:1 to IVF *clusters* for `IVF_FLAT` / `IVF_HNSW` indexes — a partition owns a contiguous
range of cluster ids.

**Test class**: `PartitionServiceTest` (`@Tag("unit")`)
- `primaryWrite_replicatesToBackups`
- `replicaVersions_monotonic`
- `reconcile_catchesUpStaleReplica`
- `primaryCrash_backupPromotedByCmg`

### 9.8 State Transfer — `StateTransferManager`

**Adopts Infinispan pattern #8**. On topology change (`ClusterEpoch` increment from CMG),
the manager computes `(source, target)` diffs and streams partition state in chunks.

```java
public interface StateTransferManager extends VectorsComponent {
    CompletionStage<Void> onTopologyChange(ClusterEpoch newEpoch);
    StateTransferProgress progress();

    record StateTransferProgress(
        int totalPartitions, int transferred, int inFlight,
        long bytesTransferred, Duration elapsed) {}
}
```

Transfers are:
- **Chunked**: 64 MiB default chunk, backpressure via reactive-streams.
- **Parallel**: N concurrent partition transfers (configurable, default 4).
- **Resumable**: per-chunk checkpoints in local WAL; interrupted transfers resume without
  re-shipping already-acked chunks.
- **Priority**: incoming primary assignments before backups.

**Test class**: `StateTransferManagerTest` (`@Tag("unit")`, in-JVM)
- `topologyChange_threePartitionsRebalanced`
- `transferInterruption_resumesFromCheckpoint`
- `chunkBackpressure_respectsDownstreamConsumer`

### 9.9 Persistence SPI — `NonBlockingStore`

**Adopts Infinispan pattern #7**. Replaces the older `StorageBackend` interface as the
canonical persistence SPI for `vectors-distributed`. The existing `StorageBackend` in
`vectors-storage` becomes a thin adapter layered on `NonBlockingStore`.

```java
public interface NonBlockingStore<K, V> {
    Set<Characteristic> characteristics();  // SHAREABLE, SEGMENTABLE, BULK_READ, TRANSACTIONAL

    CompletionStage<Void> start(StoreConfiguration cfg);
    CompletionStage<Void> stop();

    Publisher<Entry<K, V>> publishEntries(IntSet segments, Predicate<K> filter, boolean includeValues);
    CompletionStage<V> load(int segment, K key);
    CompletionStage<Void> write(int segment, Entry<K, V> entry);
    CompletionStage<Boolean> delete(int segment, K key);
    CompletionStage<Void> clear();
    CompletionStage<Long> size(IntSet segments);

    enum Characteristic { SHAREABLE, SEGMENTABLE, BULK_READ, TRANSACTIONAL, READ_ONLY }
}

/** Core implementations. External impls (S3, JDBC) live in optional modules. */
public final class LocalFileStore implements NonBlockingStore<SegmentKey, SegmentBlob> { ... }
public final class HeapStore implements NonBlockingStore<SegmentKey, SegmentBlob> { ... }
```

Optional module `vectors-distributed-s3` ships `S3Store` (AWS SDK v2 FFM-free client on
Panama-HttpClient).

**Test class**: `NonBlockingStoreContractTest` (`@ParameterizedTest`, `@Tag("unit")`)
- Runs the same test suite against every registered `NonBlockingStore` implementation.
- Validates ordering guarantees, characteristic semantics, and the `Publisher` contract.

### 9.10 Catalog & Rolling Upgrades — `CatalogManager`

**Adopts Ignite 3 pattern #12**. Every schema / collection change increments a distributed
`CatalogVersion`. The catalog is persisted in CMG (§9.6). Nodes running an older catalog
version continue to serve reads until the activation delay elapses, at which point they
switch atomically. This enables zero-downtime rolling upgrades.

```java
public interface CatalogManager extends VectorsComponent {
    CatalogVersion currentVersion();
    CollectionDescriptor collection(String name);

    /** Apply a catalog change. Waits for quorum commit; returns the new version. */
    CompletionStage<CatalogVersion> apply(CatalogChange change);

    /** Cluster-wide activation delay (typically 10s) before new version becomes effective. */
    Duration activationDelay();
}

public record CatalogVersion(long value) implements Comparable<CatalogVersion> { /* ... */ }
public sealed interface CatalogChange permits
    CatalogChange.CreateCollection, CatalogChange.DropCollection,
    CatalogChange.UpdateIndexType, CatalogChange.UpdateQuantizer { /* ... */ }
```

**Test class**: `CatalogManagerTest` (`@Tag("unit")`)
- `createCollection_bumpsVersion_allNodesConverge`
- `mixedVersionCluster_oldNodesServeOldVersionUntilActivation`
- `rollingUpgrade_sevenNodes_zeroFailedQueries` (long-running)

### 9.11 Observability — `MetricsRegistry` + OpenTelemetry

**Adopts Ignite 3 + Infinispan pattern #13**. Micrometer-based `MeterRegistry` with
Prometheus and OTLP exporters. Standard metric families: `vectors.search.latency`,
`vectors.search.recall` (when ground-truth mode enabled), `vectors.index.size`,
`vectors.wal.backlog`, `vectors.cluster.members`, `vectors.partition.rebalance.inflight`,
`vectors.gpu.utilization` (when `vectors-gpu` is present).

```java
public interface MetricsRegistry extends VectorsComponent {
    MeterRegistry meterRegistry();    // underlying Micrometer

    // Convenience recorders for hot-path metrics (pre-bound tags):
    void recordSearch(String collection, Duration latency, int resultCount);
    void recordIndexMutation(String collection, int added, int deleted);
    void recordRebalanceProgress(PartitionId partition, double fraction);
}
```

Traces follow OpenTelemetry semantic conventions. A search span wraps the BuoyIndex lookup,
`ShardRouter.plan`, each `LocalSearchRequest`, and the final top-k merge.

### 9.12 Authorization — `Subject` + `AuthorizationManager`

**Adopts Infinispan pattern #14**. JAAS-style `Subject` carried on every request context.
Collection-level ACLs stored in the catalog (§9.10). Off by default for embedded single-JVM
usage; enabled per collection via `CollectionDescriptor.authorization(...)`.

```java
public interface AuthorizationManager extends VectorsComponent {
    void authorize(Subject subject, String collection, Permission permission)
        throws SecurityException;

    enum Permission { READ, WRITE, ADMIN, SCHEMA }
}
```

### 9.13 Transport — Netty + direct-message codegen

**Adopts Ignite 3 pattern #11**. All inter-node RPC (search, replication, state transfer,
CMG) travels over a single Netty channel per peer. Messages implement a `@Transferable`
marker; annotation-processor generated `DirectMessageReader` / `DirectMessageWriter` avoid
reflection on the hot path.

```java
public interface Transport extends VectorsComponent {
    CompletionStage<Void> send(NodeId target, NetworkMessage msg);
    <R extends NetworkMessage> CompletionStage<R> request(NodeId target,
                                                          NetworkMessage req, Class<R> responseType);
    void registerHandler(Class<? extends NetworkMessage> type, MessageHandler handler);
}

@Transferable(type = 101)
public record SearchRequest(PartitionId partition, float[] query, int k) implements NetworkMessage {}
```

**Test class**: `TransportContractTest` (`@Tag("unit")`, in-JVM Netty on loopback)

### 9.14 Execution — `ShardRouter` + `ScatterGatherExecutor`

The algorithmic shell meets the deployment shell here. `BuoyIndex.route()` returns the
cluster ids for a query; `ShardRouter` maps those to partitions and thence to primary
owners via `PartitionService.owners()`. `ScatterGatherExecutor` fans out
`LocalSearchRequest`s over `Transport` and merges the results.

```java
public interface ShardRouter {
    /** From query cluster ids, produce one LocalSearchRequest per owning node. */
    List<LocalSearchRequest> plan(float[] query, int[] clusterIds, int k);
}

public final class DefaultShardRouter implements ShardRouter {
    public DefaultShardRouter(PartitionService partitions, BuoyIndex buoys);
}

/**
 * Stateless scatter-gather using virtual threads; one VT per target node per query.
 * Partial results are returned on per-node timeout (with a warning counter).
 */
public final class ScatterGatherExecutor {
    public ScatterGatherExecutor(Transport transport, Duration perNodeTimeout);
    public SearchResult execute(List<LocalSearchRequest> plan, int k);
}

public record LocalSearchRequest(
    NodeId targetNode, float[] query, int[] clusterIds, int k, float minScore) {}
```

**Test class**: `ScatterGatherExecutorTest` (`@Tag("unit")`)
- `scatterGatherAcross3Nodes_returnsTopK` — 3 in-process nodes, 10K docs each, top-10
  matches brute-force merged
- `partialTimeoutReturnsPartialResults` — one node sleeps; two nodes respond; warning logged
- `allNodesTimeOutReturnsEmpty`
- `singleNodeClusterWorksAsPassthrough`

**Benchmark class**: `ScatterGatherBenchmark` (`vectors-bench`)

| Benchmark | Nodes | N/node | D | k | Target |
|-----------|-------|--------|---|---|--------|
| `scatterGather_3nodes_FLAT` | 3 | 10K | 128 | 10 | < 5 ms p99 |
| `scatterGather_10nodes_FLAT` | 10 | 10K | 128 | 10 | < 10 ms p99 |
| `scatterGather_3nodes_HNSW` | 3 | 100K | 128 | 10 | < 20 ms p99 |

### 9.15 `VectorClusterServer` — the entry point

Embedded mode default (EclipseStore `EmbeddedStorage` + Ignite 3 `IgniteServer` pattern).
A single call brings up the entire deployment shell; auto-joins the cluster when peers
are configured, runs local-only otherwise.

```java
public interface VectorClusterServer extends AutoCloseable {
    static CompletionStage<VectorClusterServer> start(
        String nodeName, Path configPath, Path workDir);

    /** Handle to all public cluster services. */
    ClusterNode node();

    /** Access a collection as a VectorCollection — mode (LOCAL/REPL/DIST) honoured. */
    VectorCollection collection(String name);

    @Override void close();
}
```

### 9.16 `DistributedVectorCollection` — the public facade

Implements `VectorCollection` over the deployment shell. The same user-facing interface
works in `LOCAL`, `REPLICATED`, and `DISTRIBUTED` modes.

```java
public final class DistributedVectorCollection implements VectorCollection {
    // add() / delete() / upsert() → PartitionService.runOnPrimary(p, ...)
    //                             → primary appends to WAL (NonBlockingStore)
    //                             → async replicate to backups with version bump
    // search()                    → BuoyIndex.route → ShardRouter.plan → ScatterGather → merge
    // size()                      → aggregated from partition replicas (cached)
    // compact()                   → primary-driven tier compaction on each partition
}
```

**Test class**: `DistributedVectorCollectionTest` (`@Tag("unit")`, in-JVM 3-node cluster)
- `addSearchConsistency_3nodes`
- `deletePropagatesToSearchResults`
- `scatterGatherRecall_3nodes_1M_total` (`@Tag("slow")`)
- `nodeFailureReturnsPartialResults_underRf1`
- `nodeFailureNoDataLoss_underRf2` (owned-partition with backup)
- `buoyIndexUpdate_propagatesViaCmg_within500ms`

**Test class**: `DistributedVectorCollectionIT` (`@Tag("integration")`, LocalStack + Docker)
- `walAppend_persistsToS3Store_survivesColdStart`
- `coldClusterFetch_afterT2Eviction`
- `rollingUpgrade_sevenNodes_zeroFailedQueries`
- `concurrentWalWriters_noEntriesLost`
- `catalogCas_preventsDoubleSchemaCommit`

---

## 10. Integration Test Infrastructure

### 10.1 Why TestContainers + LocalStack

The pure in-JVM simulation (`InProcessNodeDirectory`, `HeapStorageBackend`) is the right
tool for unit and slow tests: it is fast, deterministic, and needs no Docker daemon.
However, T3 (cold storage) is an S3-compatible object store. The correctness of
`S3StorageBackend`, `TieredCluster.fetchFromT3`, and the distributed WAL cannot be validated
without exercising a real S3 API surface.

**TestContainers** manages ephemeral Docker container lifecycles from within JUnit 5 tests.
**LocalStack** provides a fully S3-compatible API in a single container, with no AWS
credentials, no network egress, and no billing.

Tests that require a container are tagged **`@Tag("integration")`**. They are excluded from
the default `test` task (and from `unitTest`, `slowTest`) and are invoked separately:

```bash
./gradlew integrationTest          # all modules with integration tests
./gradlew :vectors-storage:integrationTest  # single module
```

### 10.2 LocalStack Image

Starting March 2026, `localstack/localstack:latest` requires an auth token for Docker Hub
pulls. Always pin a specific release to avoid CI breakage:

```java
static final DockerImageName LOCALSTACK_IMAGE =
    DockerImageName.parse("localstack/localstack:3.8.1");
```

If your organisation has a LocalStack account, use the auth token via the env var
`LOCALSTACK_AUTH_TOKEN` and the latest image. The free Community tier is fully sufficient
for S3 — no paid features are required.

### 10.3 Gradle `integrationTest` Task

Already registered in the root `build.gradle.kts` `configure(libraryProjects)` block for
every module:

```kotlin
tasks.register<Test>("integrationTest") {
    description = "Run integration tests that require Docker (TestContainers + LocalStack)"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath    = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("integration") }
    maxParallelForks = 1   // containers are shared; avoid bucket-creation races
}
```

The default `test` task excludes `"integration"` tags, so `./gradlew build` never
requires Docker.

### 10.4 Shared Container Pattern

Start the container once per test class (static `@Container` field) to avoid paying the
~4–5 s LocalStack startup cost per test:

```java
@Tag("integration")
@Testcontainers
class S3StorageBackendTest {

    @Container
    static final LocalStackContainer LOCALSTACK =
        new LocalStackContainer(DockerImageName.parse("localstack/localstack:3.8.1"))
            .withServices(LocalStackContainer.Service.S3);

    static S3Client s3;

    @BeforeAll
    static void buildClient() {
        s3 = S3Client.builder()
            .endpointOverride(LOCALSTACK.getEndpointOverride(Service.S3))
            .credentialsProvider(StaticCredentialsProvider.create(
                AwsBasicCredentials.create(LOCALSTACK.getAccessKey(), LOCALSTACK.getSecretKey())))
            .region(Region.of(LOCALSTACK.getRegion()))
            .build();
    }

    @BeforeEach  void createBucket() { s3.createBucket(r -> r.bucket("test-vectors")); }
    @AfterEach   void deleteBucket() { /* drain and delete "test-vectors" */ }
}
```

For `TieredClusterFetchT3Test` and `DistributedVectorCollectionIT`, the same pattern
applies — a single static `LOCALSTACK` container handles both WAL writes and T3 fetches
within the same test class.

### 10.5 Which Tests Use Containers

| Test class | Module | Container service | Tag | What it validates |
|------------|--------|-------------------|-----|-------------------|
| `S3StorageBackendTest` | `vectors-storage` | LocalStack S3 | `@Tag("integration")` | `S3StorageBackend`: PutObject, GetObject, ranged GetObject, presigned URL |
| `TieredClusterFetchT3Test` | `vectors-ivf` | LocalStack S3 | `@Tag("integration")` | `TieredCluster.fetchFromT3()`: cold-cluster S3 download + mmap + search |
| `DistributedVectorCollectionIT` | `vectors-distributed` | LocalStack S3 | `@Tag("integration")` | End-to-end: WAL append → S3 → node replay → scatter-gather search |

All other test classes (`BuoyIndexTest`, `ClusterSplitterTest`, `SubBuoyTreeTest`,
`TieredClusterTest`, `ScatterGatherExecutorTest`, `DistributedVectorCollectionTest`)
use in-JVM simulation only and remain `@Tag("unit")` or untagged.

The two `DistributedVectorCollection` test classes are complementary:
- `DistributedVectorCollectionTest` (`@Tag("unit")`) — fast, in-process, no Docker; covers
  correctness of routing logic, deletion propagation, recall
- `DistributedVectorCollectionIT` (`@Tag("integration")`) — requires Docker; covers WAL
  durability, T3 fetch on cold start, S3 compare-and-swap coordination

### 10.6 Module Build Dependencies for Container Tests

The `vectors-storage/build.gradle.kts` already declares the required test dependencies.
The new `vectors-ivf` and `vectors-distributed` modules must declare the same:

```kotlin
// In vectors-ivf/build.gradle.kts and vectors-distributed/build.gradle.kts:
val testcontainersVersion = "1.21.4"
val awsSdkVersion = "2.29.52"

dependencies {
    // ... existing module deps ...
    testImplementation("org.testcontainers:testcontainers:$testcontainersVersion")
    testImplementation("org.testcontainers:junit-jupiter:$testcontainersVersion")
    testImplementation("org.testcontainers:localstack:$testcontainersVersion")
    testImplementation("software.amazon.awssdk:s3:$awsSdkVersion")
}
```

`vectors-distributed` additionally needs `software.amazon.awssdk:s3` at `implementation`
scope (not just test) for the production `S3StorageBackend`.

---

## 11. Test-First Development Sequence

The following sequence is strict: no implementation code for step N+1 before all
tests for step N pass. Each step that introduces a new concept gates the work that
depends on it.

```
──── Phase 1: IVF Foundation ────────────────────────────────────────────────────
Step P1   VectorUtilBatchTest           (vectors-core)        must pass before KMeans
Step P2   KMeansTest                    (vectors-core)        must pass before CentroidIndex
Step P3   CentroidIndexTest             (vectors-core)        must pass before BuoyIndex
Step P4   StorageBackendContractTest    (vectors-storage)     must pass before WAL
Step P5   WriteAheadLogTest             (vectors-storage)     must pass before VectorEvent
Step P6   VectorEventCodecTest          (vectors-db)          must pass before SegmentTransfer
Step P7   SegmentTransferTest           (vectors-db)          must pass before IvfIndex
Step P8   BuoyIndexTest                 (vectors-ivf)         must pass before HyperDoor
Step P9   HyperDoorTest                 (vectors-ivf)         must pass before TieredCluster
Step P10  IvfIndexIntegrationTest       (vectors-ivf)         ★ PHASE 1 GATE
          Gate: recall@10 ≥ 0.95 (N=100K, K=316, nprobe=32)
          Gate: IvfIndexBenchmark.search_N1M_K1024_D128_nprobe32 < 5 ms p99

──── Phase 2a: Hierarchical Centroid Tree (branching=100, leaf=100) ─────────────
Step P11  HierarchicalKMeansBuilderTest  (vectors-ivf)         must pass before SubBuoyTree
          Gate: recursive k=100 build terminates at leaf-size cap 100
Step P12  SubBuoyTreeTest                (vectors-ivf)         ★ PHASE 2a GATE
          Gate: concurrentReadsDuringWrite, encodeDecodeRoundTrip
          Gate: levelMemoryBudget_atDepth4_undersizedForL3
          Gate: flatTree routes identically to BuoyIndex (no regression)

──── Phase 2b: Adaptive Cluster Splitting ───────────────────────────────────────
Step P13  ClusterSplitterTest            (vectors-ivf)         ★ PHASE 2b GATE
          Gate: splitReducesCost, splitRollbackOnVerifyFail, accessFrequencyEMA
          Gate: k=100 split + merge round-trip preserves SubBuoyTree invariants

──── Phase 2c: WAL-on-S3 + Indexer/Query Role Split ─────────────────────────────
Step P14a IndexerNodeTest                (vectors-distributed) must pass before QueryNode
          Gate: consumesWalProducesIndex, manifestUpdatedViaConditionalPut
Step P14b QueryNodeTest                  (vectors-distributed) ★ PHASE 2c GATE (@Tag("unit"))
          Gate: scanUnindexedTail_returnsRecentWrites_strongConsistency
          Gate: WriteAheadLogBenchmark.throughput_10k_appends_1ns_S3 ≥ 10K a/s/ns
Step P14c WalIndexerEndToEndTest         (vectors-distributed) ★ PHASE 2c INTEGRATION GATE (@Tag("integration"))
          Gate: 10K writes, 100 indexer cycles, recall@10 unchanged

──── Phase 2d: Tiered Cascade with Confidence-Interval Rerank ───────────────────
Step P15a TieredClusterTest              (vectors-ivf)         must pass before TierPolicy
          Gate: confidenceIntervalCascade_rerankRatioBelowOnePercent
          Gate: t0T1T2Cascade recall@10 ≥ 0.98; cascadeDataRead ≤ 900 KB (50K cluster)
          Gate: t0Kernel_usesVectorPopcount_orFailsLoudly (AVX-512 CI runner)
Step P15b TierPolicyTest                 (vectors-ivf)         ★ PHASE 2d GATE
          Gate: applyAll materialises correct hot fraction (15% at θ₂=0.05)
          Gate: pinnedCluster_remainsMaterialised_evenWhenCold
          Gate: pinCap_enforced_throwsAt257thPin
          Gate: TieredClusterBenchmark.t0T1T2_N50K_D128 < 2 ms p99

──── Phase 3: Distributed Execution ─────────────────────────────────────────────
Step P16  ScatterGatherExecutorTest      (vectors-distributed) must pass before DistVC
          Gate: scatterGather_10nodes_FLAT < 10 ms p99
Step P17  DistributedVectorCollectionTest (vectors-distributed) ★ PHASE 3 GATE (@Tag("unit"))
          Gate: addSearchConsistency, deletePropagatesToSearchResults
          Gate: scatterGatherRecall_3nodes_1M_total recall@10 ≥ 0.90
          Gate: warm p50 ≤ 8 ms, cold p50 ≤ 343 ms (1M D=1024) — Turbopuffer parity
Step P18  DistributedVectorCollectionIT  (vectors-distributed) ★ PHASE 3 DURABILITY GATE (@Tag("integration"))
          Gate: walAppend_persistsToS3_survivesColdStart
          Gate: conditionalPut_preventsDoubleCommit
          Gate: coldClusterFetch_searchAfterT2Eviction
          Gate: getRange_usesHttpRangeHeader (verified in LocalStack request log)
          (Runs in CI only when Docker is available; blocks release not default build)
```

---

## 12. Phased Implementation Roadmap

> **§16 alignment**: this roadmap was reordered after the May 2026 Turbopuffer review.
> The hierarchical centroid tree (Phase 2a) precedes adaptive splitting (Phase 2b);
> a new Phase 2c lands the WAL-on-S3 layout and Indexer/Query role split BEFORE the
> tiered cascade, because the cascade's range-GET reads depend on the published
> `index/` runs being addressable.

### Phase 1 — IVF Foundation (vectors-core + vectors-storage + vectors-ivf)

**Goal**: Single-node IVF-flat search with SOAR boundary spilling, on a flat (non-adaptive)
single-level buoy index. Establishes the routing primitive and storage SPI; not yet
hierarchical and not yet tiered.

**Entry criteria**: All of P1–P9 pass (batch SIMD, KMeans, CentroidIndex, StorageBackend
including `getRange`, WAL with group commit, VectorEvent/Codec, SegmentTransfer, BuoyIndex,
HyperDoor).

**Exit gate (P10)**: `IvfIndexIntegrationTest` all pass; `IvfIndexBenchmark.search_N1M_K1024_D128_nprobe32` < 5 ms p99; `BuoyIndexBenchmark.route_K1024_D128_nprobe32` < 10 µs; `StorageBackendContractTest.getRange_*` green; `WriteAheadLogBenchmark.append_500KB_p50_S3` ≤ 285 ms (Turbopuffer parity).

| Component | Module | New files |
|-----------|--------|-----------|
| `batchDotProduct` / `batchSquaredL2` | `vectors-core` | `VectorUtil` additions |
| `KMeans` (k-means++ seeding, virtual-thread averaging) | `vectors-core` | `cluster/KMeans.java` |
| `CentroidIndex` (multi-probe + SOAR spill) | `vectors-core` | `cluster/CentroidIndex.java` |
| `StorageBackend` + `getRange` + `HeapStorageBackend` + `LocalFileStorageBackend` | `vectors-storage` | `backend/*.java` |
| `SegmentedWriteAheadLog` (CRC'd, group-commit, 512 MB cap, ◈/■ markers) | `vectors-storage` | `wal/*.java` |
| `VectorEvent` sealed hierarchy + `VectorEventCodec` | `vectors-db` | `event/*.java` |
| `SegmentExporter` / `SegmentImporter` | `vectors-db` | `transfer/*.java` |
| `BuoyIndex` (single-level root of the centroid tree, K up to 100) | `vectors-ivf` | module scaffold + `BuoyIndex.java` |
| `HyperDoor` (cross-tier O(1) ordinal link record) | `vectors-ivf` | `HyperDoor.java` |
| `ClusterPartition`, `IvfIndex` (flat IVF) | `vectors-ivf` | `ClusterPartition.java`, `IvfIndex.java` |

### Phase 2a — Hierarchical Centroid Tree (vectors-ivf)

**Goal**: The single-level `BuoyIndex` is generalised into a multi-level `SubBuoyTree`
with **branching factor 100** and **leaf target ~100 vectors**. Routing for collections
beyond N≈10K descends through the tree instead of doing a flat scan over the root.
The tree is built deterministically by recursive k-means at build time (no adaptive
behaviour yet — that's Phase 2b).

**Entry criteria**: Phase 1 exit gate passes; P10 (IvfIndexIntegrationTest) all green.

**Exit gate (P12)**: `SubBuoyTreeTest` all pass; `levelMemoryBudget_atDepth4_undersizedForL3`
green; `IvfIndexBenchmark.search_N100M_branching100_depth3` recall@10 ≥ 0.90 and p99 ≤ 50 ms
single-node.

| Component | Module | New files |
|-----------|--------|-----------|
| `SubBuoyTree` (recursive tree, branching=100, gossip-serialisable) | `vectors-ivf` | `cluster/SubBuoyTree.java` |
| `HierarchicalKMeansBuilder` (recursive k=100 build, leaf-size cap 100) | `vectors-ivf` | `cluster/HierarchicalKMeansBuilder.java` |

### Phase 2b — Adaptive Cluster Splitting (vectors-ivf)

**Goal**: The hierarchical tree gains the runtime split/merge behaviour driven by the Quake
cost model. Recall improves for skewed workloads without rebuilding the tree from scratch.

**Entry criteria**: Phase 2a exit gate passes.

**Exit gate (P13)**: `ClusterSplitterTest` all pass; `SubBuoyTreeTest.concurrentReadsDuringWrite`
green; no regression on `IvfIndexIntegrationTest`.

| Component | Module | New files |
|-----------|--------|-----------|
| `ClusterSplitter` (Quake cost model, k=100 split, merge) | `vectors-ivf` | `cluster/ClusterSplitter.java` |

### Phase 2c — WAL-on-S3 + Indexer/Query Role Split (vectors-distributed)

**Goal**: Implement the Turbopuffer write contract end-to-end: writes append to
`s3://{bucket}/{ns}/wal/NNN.log` via group commit; a background `IndexerNode` consumes
closed `wal/` objects and produces immutable runs in `s3://{bucket}/{ns}/index/`; a
`QueryNode` reads `index/` and scans the unindexed `wal/` tail for strong-consistency
queries. Both roles can run in the same JVM (embedded mode) or separate processes.

**Entry criteria**: Phase 2a exit gate passes (Phase 2b is parallel; not a hard prereq).

**Exit gate (P14b)**: `IndexerNodeTest.consumesWalProducesIndex` green;
`QueryNodeTest.scanUnindexedTail_returnsRecentWrites_strongConsistency` green;
`WalIndexerEndToEndTest` (P14c, LocalStack: 10 000 writes, 100 indexer cycles,
recall@10 unchanged) green; `WriteAheadLogBenchmark.throughput_10k_appends_1ns_S3`
≥ 10 000 a/s/ns.

| Component | Module | New files |
|-----------|--------|-----------|
| `IndexerNode` (background daemon: WAL → index runs) | `vectors-distributed` | `indexer/IndexerNode.java` |
| `QueryNode` role (scans `index/` + ◈ tail) | `vectors-distributed` | `query/QueryNode.java` |
| `NamespaceLayout` (resolves `{ns}/wal/`, `{ns}/index/`, `manifest.json`) | `vectors-storage` | `wal/NamespaceLayout.java` |
| `S3StorageBackend` (with `getRange` via `Range:` header) | `vectors-distributed` | `backend/S3StorageBackend.java` |

### Phase 2d — Tiered Cascade with Confidence-Interval Rerank (vectors-ivf)

**Goal**: Per-cluster four-tier cascade (T0=1-bit/heap, T1=SQ8/off-heap, T2=float32/mmap,
T3=float32/S3) driven by `TierPolicy` with **RaBitQ confidence-interval rerank**. Rerank
ratio < 1 % at recall@10 ≥ 0.95 (Phase 1 gate); ≥ 0.98 with full T0→T1→T2 (Phase 2 gate).
Pinning override exposed via `TierPolicy.pin(clusterId, tier)`.

**Entry criteria**: Phase 2c exit gate passes (range-GET available end-to-end).

**Exit gate (P15b)**: `TieredClusterTest` (P15a) all pass including
`confidenceIntervalCascade_rerankRatioBelowOnePercent`; `TierPolicyTest.applyAll` correct;
`TierPolicyTest.pinnedCluster_remainsMaterialised_evenWhenCold` green;
`TierPolicyTest.pinCap_enforced_throwsAt257thPin` green;
`TieredClusterBenchmark.t0T1T2_N50K_D128` < 2 ms p99; `cascadeDataReadBelow1MB_50KCluster`
≤ 900 KB.

| Component | Module | New files |
|-----------|--------|-----------|
| `TieredCluster` (confidence-interval cascade, fetchOrdinalFromT3) | `vectors-ivf` | `tier/TieredCluster.java` |
| `TierPolicy` (θ₁/θ₂ + pin/unpin, max 256 pins) | `vectors-ivf` | `tier/TierPolicy.java` |
| `IvfHnswIndex` (graph per cluster, `IndexType.IVF_HNSW`) | `vectors-ivf` | `IvfHnswIndex.java` |

### Phase 3 — In-Process Distributed Execution (vectors-distributed)

**Goal**: Multi-node in-process cluster using `InProcessNodeDirectory`; all
`DistributedVectorCollectionTest` pass. Distribution above 2 TB / 500M vectors per
namespace uses **random sharding** (vector → shard via `id % N`), **broadcast** of the
ANN query to all shards, and **global top-k merge** (Turbopuffer §16.6). No real network
required. Production wiring (gRPC, real S3) is Phase 4.

**Entry criteria**: Phase 2d exit gate passes.

**Exit gate (P17)**: `ScatterGatherExecutorTest` (P16) green;
`DistributedVectorCollectionTest.scatterGatherRecall_3nodes_1M_total` recall@10 ≥ 0.90;
`ScatterGatherBenchmark.scatterGather_10nodes_FLAT` < 10 ms p99;
**warm latency target**: p50 ≤ 8 ms / p90 ≤ 10 ms for 1M D=1024 vectors (Turbopuffer parity);
**cold latency target**: p50 ≤ 343 ms / p90 ≤ 444 ms;
**P18 (integration)**: `DistributedVectorCollectionIT` durability tests green
(see §11 step P18).

| Component | Module | New files |
|-----------|--------|-----------|
| `NodeId`, `ShardOwnership` (random hash for >2 TB namespaces, consistent hash for ≤ 2 TB), `ShardRouter` | `vectors-distributed` | module scaffold |
| `ScatterGatherExecutor` (virtual threads) + `InProcessNodeDirectory` | `vectors-distributed` | `ScatterGatherExecutor.java` |
| `StaticClusterMembership` + `GossipMembership` | `vectors-distributed` | `membership/*.java` |
| `DistributedVectorCollection` (implements `VectorCollection`) | `vectors-distributed` | `DistributedVectorCollection.java` |

### Phase 4 — Network Transport (Future)

gRPC service definitions, TLS, node discovery via DNS-SRV or Kubernetes endpoints.
Deferred until Phase 3 is stable and a concrete deployment target is identified.

---

## 13. Cost Analysis — 100M Vector Cluster

Working set model for a 100M-vector, 128-dimension deployment on a 3-node cluster.

### Memory and Storage Breakdown

| Tier | Representation | Total size | Materialised fraction | RAM per node |
|------|----------------|------------|----------------------|--------------|
| T0 | 1-bit RaBitQ | 1.6 GB | 100% (all nodes) | ~533 MB/node |
| T1 | SQ8 off-heap | 12.8 GB | 15% hot clusters | ~640 MB/node |
| T2 | float32 mmap | 51.2 GB | 5% hottest clusters | ~853 MB/node (OS page cache) |
| T3 | float32 S3 | 51.2 GB | always (source of truth) | — (object store) |

**Total RAM per node** (T0 global + T1+T2 working set, 3-node cluster): ~2 GB active + ~3 GB page cache headroom. A single node with 16 GB heap serves 100M vectors comfortably.

### Search Latency Budget (single-node path)

```
BuoyIndex.route (Layer 1)     <    1 µs   SIMD dot, 524 KB in L2 cache
SubBuoyTree.refineToLeaves    <    5 µs   heap traversal, depth ≤ 3
TieredCluster.search T0 pass  <  800 µs   1-bit, 50K ordinals × 128 dims
TieredCluster.search T1 pass  <  128 µs   SQ8, 1K survivors × 128 dims
TieredCluster.search T2 pass  <   51 µs   float32, 100 survivors × 512 bytes
                               ─────────
Total (nprobe=32, hot cluster) < 2 ms p99  single-node
Scatter-gather overhead       < 5 ms add   10-node cluster, virtual threads
Total distributed p99         < 7 ms       well within 10 ms target
```

### S3 Cost (at AWS us-east-1 pricing, April 2026)

| Item | Calculation | Monthly cost |
|------|-------------|--------------|
| T3 storage (51.2 GB, 3× replication) | 153.6 GB × $0.023/GB | $3.53 |
| S3 GET (cold cluster fetch, 1000/day) | 30K × $0.0004/1K | $0.01 |
| S3 PUT (WAL flush, 100/day) | 3K × $0.005/1K | $0.02 |
| **Total S3** | | **~$3.56/month** |

At 1 billion vectors (D=128): T3 = 512 GB raw → ~$35/month S3. T0 (1-bit) = 16 GB,
fits on every node's heap. Cluster RAM requirements scale linearly with the hot fraction only.

### I/O Read Reduction vs Brute-Force

```
Brute-force (float32, 1M vectors, D=128): 512 MB per query
IVF-flat nprobe=32 (32 clusters × ~1K vecs): 16 MB per query — 32× reduction
TieredCluster T0→T1→T2 (50K cluster):        ~980 KB per cluster — 26× vs brute-force
Combined (IVF + cascade):                     <1 MB per query — 512× vs brute-force
```

---

## 14. Open Questions

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

4. **Replication factor**: RF>1 is handled by the owned-partition + versioned-replica
   pattern (§9.7 — Hazelcast). Writes go to the primary synchronously and fan out async to
   backups; per-replica version counters drive anti-entropy reconciliation. RF=1 is the
   degenerate case of the same code path. No separate protocol is needed — **this question
   is resolved by the §11 pattern adoption**.

5. **Filter + IVF interaction**: Pre-filtering (restrict ANN search to matching ordinals per
   cluster) vs post-filtering (filter after ANN, existing behaviour). Pre-filtering within
   `ClusterPartition` enables better recall for selective filters but requires the filter
   executor to also be invoked at the cluster level.

6. **Quantizer-per-cluster vs global**: FAISS trains one PQ codebook per cluster for OPQ
   alignment. RaBitQ uses one global rotation. Recommendation: global RaBitQ for T0 (already
   implemented); per-cluster SQ8 for T1 (cluster distribution centred, better utilisation).

---

## 15. Mapping to `competitive-analysis.md` §11 patterns

Every pattern catalogued in `competitive-analysis.md` §11 maps to a concrete component of
this design. The table is the contract: a pattern listed there must correspond to a named
section or class below; new patterns adopted later must update both documents.

| §11 # | Pattern | Source reference | This design — section & class |
|---|---|---|---|
| 1 | Embedded-first lifecycle (`start(name, config, workDir)`, auto-join or local-only) | EclipseStore `EmbeddedStorage` + Ignite 3 `IgniteServer` | §9.15 `VectorClusterServer` |
| 2 | `LifecycleManager` DAG, reverse shutdown | Ignite 3 `LifecycleManager` | §9.3 `VectorsComponent` + `LifecycleManager` |
| 3 | Gossip + Raft hybrid (SWIM for liveness, Raft for metadata) | Ignite 3 `ScaleCubeClusterService` + `ClusterManagementGroupManager` | §9.5 `SwimClusterService` + §9.6 `ClusterManagementGroup` |
| 4 | Owned-partition + versioned replicas | Hazelcast `InternalPartitionServiceImpl` + `PartitionReplicaVersionManager` | §9.7 `PartitionService` + `PartitionReplicaVersionManager` |
| 5 | `DiscoveryStrategy` SPI (multicast, static, k8s, consul) | Hazelcast `com.hazelcast.spi.discovery` | §9.4 `DiscoveryStrategy` |
| 6 | `CacheMode` (LOCAL / REPL / DIST) switch on builder | Infinispan `CacheMode` | §3.4 deployment topologies + `CollectionBuilder.mode(...)` in `vectors-db` |
| 7 | `NonBlockingStore<K,V>` SPI with characteristics | Infinispan `NonBlockingStore` | §9.9 `NonBlockingStore` + `LocalFileStore` / `HeapStore` / `S3Store` |
| 8 | Segment-based state transfer (chunked, resumable) | Infinispan `StateTransferManager` + Ignite 3 | §9.8 `StateTransferManager` |
| 9 | Lazy `Root<T>` + `Lazy<T>` for large graphs | EclipseStore `RootReference` + `Lazy<T>` | §9.9 `NonBlockingStore.load(segment, key)` + T3 fetch in `TieredCluster.fetchFromT3` |
| 10 | Housekeeping / consolidation background task | EclipseStore `StorageHousekeeping` + Chronicle | `TieredCluster.compact()` (driven by `PartitionService` primary) + WAL truncation in `NonBlockingStore` |
| 11 | Direct message codegen via `@Transferable` | Ignite 3 `DirectMessageReader` / `DirectMessageWriter` | §9.13 `Transport` + `@Transferable` annotation processor |
| 12 | Catalog versioning for rolling upgrades | Ignite 3 `CatalogManager` + `UpdateLog` | §9.10 `CatalogManager` + `CatalogVersion` + CMG-backed `UpdateLog` |
| 13 | Micrometer metrics + OTel traces | Infinispan `MetricsRegistry` + Ignite 3 `OtlpMetricsExporter` | §9.11 `MetricsRegistry` |
| 14 | JAAS `Subject` + `AuthorizationManager` | Infinispan `SecureCacheImpl` | §9.12 `AuthorizationManager` + `Subject` in request context |
| 15 | Multi-release JAR (JDK 21 fallback, JDK 25 Vector API) | `cuvs-java` multi-release `pom.xml` | `vectors-distributed/build.gradle.kts` multi-release JAR packaging (Phase 4) |

### 15.1 What is NOT adopted (and why)

| Not adopted | Why |
|---|---|
| Full Raft per partition (etcd / CockroachDB model) | Our data path writes are latency-sensitive; CMG-in-Raft + async replication is the Ignite 3 / Hazelcast compromise that keeps consensus off the hot path. |
| Compute grid (`EntryProcessor` running arbitrary user code on owner) | Non-goal. A vector DB's domain is search, not arbitrary compute. We expose scatter-gather, not general distributed tasks. |
| JPA / SQL front-end (Infinispan HotRod / Ignite SQL) | Non-goal. The `VectorCollection` API and its Spring AI / LangChain4j adapters are the surface. Optional `vectors-grpc` front-end is tracked separately. |
| Off-heap object graph allocator (Chronicle Map `Byteable` codegen) | Our off-heap story uses `MemorySegment` + `Arena` directly (see `vectors-storage`). Chronicle's codegen solves a different problem (replacing a `HashMap` drop-in). |
| Kubernetes operator in-tree | Out of scope for the core. Can ship as `vectors-distributed-k8s` optional module once GA. |

### 15.2 Validation rule

Before a `vectors-distributed` PR is merged, the author confirms that every new cluster
component either:

1. Appears in the table above (pattern # cited in Javadoc), **or**
2. Is added to the table in the same PR, with a companion entry in
   `competitive-analysis.md` §11 identifying the source reference repo and a justification.

This keeps the two documents in sync as the design evolves and ensures every cluster-layer
decision is anchored to a pattern that has shipped at scale.

---

## 16. Turbopuffer alignment (May 2026 revision)

This section is the canonical mapping between Turbopuffer's published architecture
(`turbopuffer.com/docs/architecture`, `turbopuffer.com/blog/ann-v3`, `docs/limits`,
`docs/performance`, May 2026) and the `vectors-distributed` design. It supersedes any
conflicting text in earlier sections — the interleaved revisions in §3.1, §6.1, §6.2,
§8.5, §8.7, §8.8, and §12 already reflect what is here.

The mapping is presented as a sequence of **invariants we adopt verbatim**, each with the
specific numeric contract and the section/class in this design that owns it.

### 16.1 Production scale and latency contract

| Metric | Turbopuffer (production, May 2026) | This design — target | Owning section |
|---|---|---|---|
| Vectors per namespace | 500M @ 2 TB (hard cap) | 500M @ 2 TB before split | §16.6 |
| Writes per namespace | 10 000/s @ 32 MB/s | 10 000 appends/s/namespace | §6.2 group commit |
| Queries per namespace | 1 000+/s | 1 000+/s with NVMe-warm | §3.3 query path |
| Pinned namespaces / account | 256 | 256 (`TierPolicy.maxPins`) | §8.8 |
| Cold latency, 1M D=1024 vectors | p50 ≈ 343 ms, p90 ≈ 444 ms | p50 ≤ 343 ms, p90 ≤ 444 ms | §12 P17 gate |
| Warm latency, 1M D=1024 vectors | p50 ≈ 8 ms, p90 ≈ 10 ms | p50 ≤ 8 ms, p90 ≤ 10 ms | §12 P17 gate |
| Cold latency, 1M docs BM25 | p90 ≈ 285 ms | (full-text out of scope; see Non-Goals) | n/a |
| WAL group-commit p50, 500 kB | ≈ 285 ms | ≤ 285 ms (§6.2 benchmark) | §6.2 |
| Recall@10 | 90–95 % | ≥ 0.90 (Phase 1), ≥ 0.95 (Phase 2d) | §12 |

These numbers are the **acceptance contract** for the distributed implementation. JMH
targets in §6.2, §8.5, §8.7 are pinned to them; regressions block phase exits.

### 16.2 S3 namespace layout and write contract

```
s3://{bucket}/{namespace_id}/
    wal/
        000000000001.log     ■ closed, indexed
        000000000002.log     ■ closed, indexed
        000000000003.log     ◈ closed, NOT YET indexed (query nodes scan this)
        000000000004.log        active group-commit buffer (flushed every 1 s)
    index/
        000000000001.idx     immutable run produced by IndexerNode
        000000000002.idx
    manifest.json            JSON: list of ■ vs ◈ wal/* + current index epoch
```

**Write contract** (`§6.2 WriteAheadLog`):

1. Every `append(entry)` is buffered in memory by the per-namespace writer.
2. A 1-second tick flushes the buffer to a single `wal/NNNNNNNNNN.log` object via
   `StorageBackend.put` (group commit). Concurrent `append` callers awaiting the flush
   are released only after the PUT acks.
3. Each WAL object is capped at **512 MB**; an append that would overflow forces an
   early flush + new object.
4. The new object is a ◈ entry until `IndexerNode` consumes it and calls
   `markIndexed`, after which `manifest.json` is rewritten via `conditionalPut`.
5. Query consistency:
   - **Strong consistency** (default): query scans `index/` ∪ ◈-tail of `wal/`.
   - **Eventual consistency** (opt-in, ≤ 1 hour staleness): query scans only `index/`.

This contract is what makes the durability story "object storage is the only stateful
dependency" — there is no broker, no Raft on the data path. CMG (§9.6) covers
**metadata** consensus only.

### 16.3 ANN v3 hierarchical centroid tree

Adopted verbatim from Turbopuffer ANN v3 (May 2026 blog):

| Parameter | Value | Rationale |
|---|---|---|
| Branching factor | **100** between adjacent levels | matches DRAM/SSD size ratio (10×–50×) |
| Leaf cluster size | **~100 vectors** | balances tree depth with per-leaf scan cost |
| Probe budget per query | **~500 leaves** | empirical 90–95 % recall@10 for D=1024 |
| Centroid level residency | L1 → L2 → L3 → DRAM (top-down) | tree shape mirrors memory hierarchy |
| Bandwidth per level (RaBitQ) | ~6 MB/level | 16× compression vs. f16 (100 MB/level) |

Concrete shapes for collections of various sizes (D=1024, 1-bit RaBitQ centroids):

| N (vectors) | Tree depth | Levels | Leaves | Total centroid bytes |
|---|---|---|---|---|
| 10K | 2 | root only | 100 | 12.8 KB |
| 1M | 3 | root + 1 | 10K | 1.28 MB |
| 100M | 4 | root + 2 | 1M | 128 MB |
| 10B | 5 | root + 3 | 100M | 12.8 GB |

The tree is owned by `BuoyIndex` (root level only — replicated to every node) plus
`SubBuoyTree` (deeper levels — heap-resident on each node where the underlying leaves
are owned). Build is hierarchical k-means with k=100 at every level; runtime split/merge
is the `ClusterSplitter` Quake cost model (§8.4).

### 16.4 RaBitQ with confidence-interval rerank

Adopted verbatim. The T0 1-bit kernel returns a distance **interval** `[lo, hi]` per
ordinal; only ordinals whose interval straddles the running top-k threshold τ_k are
reranked at higher precision. The empirical contract:

- **Rerank ratio: < 1 %** of T0 candidates ever reach T1/T2/T3.
- **Recall@10 ≥ 0.95** with full T0→T1→T2 cascade (Phase 2 gate).
- **Inner-loop bandwidth:** ~6 MB at T0 + ~64 KB at T1 + ~25 KB at T2 per cluster
  (50K-vector cluster, D=128) → ~890 KB total, vs. 25.6 MB brute-force float32
  (29× reduction).

The implementation is owned by `TieredCluster.search` (§8.7). The `rerankRatioCap`
parameter on `CascadeParams` is the policy knob: exceeding the cap triggers a
recall-preserving widening of τ_k rather than a bandwidth blowout.

### 16.5 Pinning

`TierPolicy.pin(clusterId, Tier)` (§8.8) is the explicit override on top of the
EMA-driven θ₁/θ₂ thresholds. Pinned clusters are force-materialised and protected from
eviction. The cap is **256 pins per collection** (Turbopuffer parity). This is the
operator-facing knob for latency-sensitive tenants who do not want a cold start after
inactivity.


### 16.6 Distribution above the per-namespace cap

Single-namespace cap: **2 TB / 500M vectors**. Above that, follow Turbopuffer's
explicit recommendation: **random sharding + broadcast + global merge**.

```
write(id, vec):
    shard_id = hash(id) % N
    route to namespace shard_id

query(q, k):
    fanout = scatter q to all N namespace shards (virtual threads)
    per-shard top-k = local ANN search on that shard
    global top-k = merge of N shard results (heap merge, O(N · k · log k))
```

This is intentionally boring: no smart partitioning, no consistent hashing of vectors,
no co-locating "similar" vectors. The justification (Turbopuffer ANN v3 blog):

> "It is crucial to maximize the efficiency of a single machine before turning to
> distribution. Moving in the other direction leads to an unnecessarily expensive
> system."

This also means a single-namespace deployment is the optimisation target up to 500M
vectors, and the distributed shell only kicks in above that. The `CacheMode.DISTRIBUTED`
builder flag (§3.4) selects this mode; routing is owned by `ShardRouter` (§9.14) with
random hashing for namespaces above the cap.

### 16.7 Compute-bound inner loop and JDK-25 SIMD plan

Once T0 is 1-bit-quantised, the workload is **compute-bound**, not bandwidth-bound. Each
fetched bit is reused ~4× by the RaBitQ kernel (RaBitQ paper §3.3.2 — distance estimation
reuses `BIT(d_i, q_i) ⊕ BIT(d_i, q_{i+1})` etc.), giving a 64× higher arithmetic intensity
than `f16` distance kernels.

| Hardware capability | T0 kernel target | Java path |
|---|---|---|
| AVX-512 with `VPOPCNTDQ` | 1 instruction/cycle pipelined | `LongVector.bitCount()` lowers to `VPOPCNTDQ` on JDK-25 |
| AVX-512 without `VPOPCNTDQ` | ~3× slower (bit-shift + popcnt) | C2 falls back automatically; metric flagged |
| AVX2 only | ~2× slower (4-bit-table popcount) | C2 falls back; metric flagged |
| ARMv8 NEON | competitive (`CNT` instruction available) | `LongVector.bitCount()` lowers to `CNT` |

Verification protocol:

1. JMH microbenchmark `BitDistanceKernelBenchmark.popcount_throughput` measures
   gigabits/sec achieved against a 100M 1-bit-vector blob.
2. `-XX:+UnlockDiagnosticVMOptions -XX:+PrintAssembly` snapshot is captured in CI on
   AVX-512 hardware; the assertion is "the inner loop contains `VPOPCNTDQ` (or `CNT`
   on ARM) and not the table-based fallback."
3. Startup detects the SIMD capability and exposes it as the
   `vectors.simd.popcount.path` metric (`avx512_vpopcntdq` / `avx512_fallback` /
   `avx2_fallback` / `neon_cnt` / `scalar`). Production deployments alarm on the
   non-`vpopcntdq` paths.
4. The `t0Kernel_usesVectorPopcount_orFailsLoudly` test in `TieredClusterTest` is the
   regression guard: it only runs (`assumeTrue`) on AVX-512 hardware but is mandatory
   in CI on a labelled runner.

This is the contract that lets the JVM kernel approach Turbopuffer's reported
~10 000 qps/node ceiling on production hardware. Falling off the `VPOPCNTDQ` path is
a cliff, not a slope — sizing decisions assume the fast path.

### 16.8 Cost waterfall

For documentation parity, the storage-cost waterfall Turbopuffer publishes (and which
this design preserves):

| Architecture | $ / TB / month |
|---|---|
| RAM + 3× SSD (incumbent vector DBs) | $3,600 |
| RAM cache + 3× SSD (relational DBs) | $1,600 |
| 3× SSD only | $600 |
| **S3 + SSD cache (this design at `CacheMode.DISTRIBUTED`)** | **~$70** |
| S3 only (cold-only / archival) | $20 |

The §13 cost analysis estimates ~$3.56/month for a 100M-vector D=128 deployment, which
is consistent with the $70/TB/month line item once you fold in NVMe cache cost. The
relevant point is the ~50× delta vs. in-memory vector DBs and the ~9× delta vs. 3×SSD —
this is the value proposition that justifies the entire architecture.

### 16.9 Cross-references and conflict-resolution rules

When earlier sections of this document conflict with §16, **§16 wins**. The interleaved
revisions below are already aligned; this list is for future authors who add or change
text in the affected sections.

| Topic | §16 sub-section | Aligned section(s) | Symbol/file |
|---|---|---|---|
| Hierarchical centroid tree, branching=100, leaf=100 | §16.3 | §3.1, §8.5 | `BuoyIndex`, `SubBuoyTree` |
| Range-GET on `StorageBackend` | §16.2 | §6.1 | `StorageBackend.getRange` |
| WAL group commit + namespace prefix layout | §16.2 | §6.2, §3.4 | `SegmentedWriteAheadLog`, `NamespaceLayout` |
| RaBitQ confidence-interval rerank, < 1 % rerank ratio | §16.4 | §3.1, §8.7 | `TieredCluster.search`, `CascadeParams.rerankRatioCap` |
| Pinning override on `TierPolicy` | §16.5 | §8.8 | `TierPolicy.pin/unpin/maxPins` |
| Random-shard + broadcast above 500M/2 TB | §16.6 | §3.4, §9.14, §12 P17 | `ShardRouter`, `CacheMode.DISTRIBUTED` |
| Compute-bound T0 + `VPOPCNTDQ` lowering | §16.7 | §3.1, §8.7 | `BitDistanceKernel`, `vectors.simd.popcount.path` metric |
| 500M/2 TB hard caps, 256 pins, 10k w/s, 1k q/s | §16.1 | §12 phase exit gates | JMH `*Benchmark` targets |

**Validation rule** (companion to §15.2): a PR that touches §3.1, §6.1, §6.2, §8.5, §8.7,
§8.8, or §12 must either preserve the §16 contract or update §16 in the same PR. CI
parses the §16.1 latency table and refuses a phase-exit-gate change that loosens any
target without an accompanying §16 entry.
