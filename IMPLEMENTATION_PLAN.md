# java-vectors — Implementation Plan

> Last updated: 2026-04-15  
> Strategy: dependency-topological order; API-breaking changes before additive ones;
> TDD gate (Red → Green) before each merge.

---

## Current State Snapshot

| Module | Status | Notes |
|---|---|---|
| vectors-core | ✅ Complete | SIMD kernels, KMeans++, CentroidIndex |
| vectors-storage | ✅ Complete | StorageBackend, SegmentedWriteAheadLog, MappedVectorStore, SlabAllocator |
| vectors-quantization | ✅ Complete | SQ8/4, BQ, RaBitQ, ExtRaBitQ, PQ, NVQ, TurboQuant + ScoreFunction |
| vectors-hnsw | ✅ Complete | HnswIndex, builder, searcher, two-pass quantized search |
| vectors-vamana | ✅ Complete | VamanaIndex, builder, searcher, two-pass quantized search |
| vectors-ivf | ✅ Complete | IvfIndex, BuoyIndex, SubBuoyTree, ClusterSplitter, TieredCluster, TierPolicy, DistributedVectorCollection (WAL replay) |
| vectors-db | ⚠️ Gap | VectorCollection complete for FLAT/HNSW/VAMANA; `IVF_FLAT` throws UnsupportedOperationException (lines 319, 390 of VectorCollectionImpl) |
| vectors-bench | ❌ Missing | Dataset loaders ✅, RecallUtil ✅, slow-test recall guards ✅ — but **zero @Benchmark classes** |

**43 tests passing. Full build green.**

---

## Gaps (concrete, confirmed)

1. **`IVF_FLAT` in `VectorCollectionImpl`** — two `UnsupportedOperationException` switch arms  
   (in-memory build path line 319, mapped reopen path line 390).
2. **JMH benchmark classes** — Tier 1 (kernel), Tier 2 (build), Tier 3 (ANN recall-QPS), Facade — all missing.
3. **`S3StorageBackend`** — `DistributedVectorCollectionIT` uses `HeapStorageBackend` as a stand-in.
4. **`HyperDoor` cross-tier wiring** — exists as a record; not integrated into `TieredCluster` eviction decisions.

---

## Ordering Rationale (why this sequence minimises backtracking)

```
Phase 4 (IVF_FLAT adapter)
  └─ modifies VectorCollectionImpl once — before benchmarks measure it
Phase 5 (JMH Tier 1 — kernel microbenchmarks)
  └─ pure additions to vectors-bench; no library changes
Phase 6 (JMH Tier 2 — build benchmarks)
  └─ same; adds IVF_FLAT build benchmark after Phase 4 completes it
Phase 7 (JMH Tier 3 — ANN recall-QPS + facade)
  └─ same; covers all four index types only after IVF_FLAT exists
Phase 8 (S3StorageBackend + LocalStack IT)
  └─ additive to vectors-storage; no changes to any other module
Phase 9 (HyperDoor cross-tier integration)
  └─ additive polish; depends on TieredCluster (done) and S3 backend (Phase 8)
```

---

## Phase 4 — IVF_FLAT Adapter in `vectors-db`

**Goal**: all four `IndexType` values work end-to-end in `VectorCollection`.  
**Gate test**: `P18 — VectorCollectionIvfFlatTest` (`@Tag("unit")`)

### 4a — `IvfFlatAdapter` (in-memory)

**Create** `vectors-db/.../index/IvfFlatAdapter.java`:
- Implements `IndexSpi`
- Wraps `IvfIndex` (from `vectors-ivf`)
- `build(float[][] vectors, SimilarityFunction metric)` → delegates to `IvfIndex.build(..., IvfBuildParams)`
- `search(float[] query, int k, Predicate<Integer> filter)` → `IvfIndex.search(IvfSearchRequest)`, applies filter post-scan, maps `IvfHit` → `IndexSpi.SearchOutcome`
- Parameters: `k` (clusters), `nprobe`, `gamma` exposed via `IvfFlatAdapterConfig` (or reuse `IvfBuildParams`)

**Modify** `vectors-db/build.gradle.kts`: add `api(project(":vectors-ivf"))`

### 4b — `MappedIvfFlatAdapter` (persistent/read-only)

**Create** `vectors-db/.../storage/MappedIvfFlatAdapter.java`:
- Implements `IndexSpi` (read-only; `build()` throws `UnsupportedOperationException` per MappedHnsw contract)
- Deserialises `IvfIndex` from `GenerationDirectory` via a new `IvfIndexCodec`
- `IvfIndexCodec`: serialise centroid float[][] + per-cluster ordinal int[] to a `byte[]` blob; store under `"ivf-index"` key in `GenerationDirectory`

### 4c — Wire into `VectorCollectionImpl`

**Modify** `VectorCollectionImpl` (two switch arms):
- in-memory build path (line 319): `case IVF_FLAT -> buildIvfFlatAdapter(config)`
- mapped reopen path (line 390): `case IVF_FLAT -> new MappedIvfFlatAdapter(dir, config)`

### 4d — Gate test (write first, TDD)

**Create** `vectors-db/.../VectorCollectionIvfFlatTest.java` (`@Tag("unit")`):
- `buildAndSearch_roundTrip` — add 500 vectors, search, recall ≥ 0.80
- `commit_thenReopen_preservesVectors` — close + reopen via `VectorCollectionBuilder.open()`
- `search_appliesFilter` — metadata filter reduces result set
- `search_withDelete_excludesTombstoned` — deleted doc not returned
- `sizeReflectsAdds` — size() increments correctly

**Modify** `vectors-db/build.gradle.kts`: add `testImplementation(project(":vectors-ivf"))`

---

## Phase 5 — JMH Tier 1: Kernel Microbenchmarks

**Goal**: validate SIMD speedup; catch scalar-vs-Panama regressions.  
**Gate**: `./gradlew :vectors-bench:jmh -Pjmh.includes=".*Kernel.*"` produces QPS numbers.

### 5a — `DistanceKernelBenchmark`

**Create** `vectors-bench/.../bench/DistanceKernelBenchmark.java`:
- `@Param({"1","4","8","16","32","64","96","128","256","384","512","768","1024","1536"})` dim
- `@Param({"true","false"}) boolean usePanama` — forces provider via `VectorizationProvider`
- `@Benchmark dotProduct`, `squareDistance`, `cosine`
- `@BenchmarkMode(Throughput)`, `@OutputTimeUnit(MICROSECONDS)`
- `@Warmup(iterations=4, time=1s)`, `@Measurement(iterations=5, time=1s)`, `@Fork(3)`

### 5b — `QuantizerKernelBenchmark`

**Create** `vectors-bench/.../bench/QuantizerKernelBenchmark.java`:
- SQ8 encode throughput; `@Param({"128","768","1536"})` dim
- One benchmark per quantizer kind (SQ8, BQ, RaBitQ)

---

## Phase 6 — JMH Tier 2: Build Benchmarks

**Goal**: construction-time baselines before any optimisation work begins.  
**Gate**: `./gradlew :vectors-bench:jmh -Pjmh.includes=".*Build.*"` runs without error.

### 6a — `HnswIndexBuildBenchmark`

**Create** `vectors-bench/.../bench/HnswIndexBuildBenchmark.java`:
- `@Param({"10000","100000"})` N, `@Param({"32","64","128"})` R, `@Param({"0","1"})` useSQ8
- `@BenchmarkMode(AverageTime)`, `@OutputTimeUnit(SECONDS)`
- `@Setup(Level.Trial)` generates random dataset; `@Benchmark` builds full index

### 6b — `VamanaIndexBuildBenchmark`

**Create** `vectors-bench/.../bench/VamanaIndexBuildBenchmark.java`:
- Same parameter space as HNSW build benchmark

### 6c — `IvfIndexBuildBenchmark` (only after Phase 4 complete)

**Create** `vectors-bench/.../bench/IvfIndexBuildBenchmark.java`:
- `@Param({"4","8","16","32"})` K (cluster count), `@Param({"10000","100000"})` N

### 6d — `ScalarQuantizerTrainBenchmark`

**Create** `vectors-bench/.../bench/ScalarQuantizerTrainBenchmark.java`:
- `@Param({"128","768","1536"})` dim, `@Param({"10000","100000"})` N

---

## Phase 7 — JMH Tier 3: ANN Recall-QPS + Facade

**Goal**: recall@k vs QPS tradeoff curves on SIFT Small and (optionally) SIFT 1M.  
**Gate**: `SiftSmallHnswRecallTest` and `SiftSmallVamanaRecallTest` (already `@Tag("slow")`) pass.

### 7a — `HnswSearchBenchmark`

**Create** `vectors-bench/.../bench/HnswSearchBenchmark.java`:
- `@Param({"10","20","50","100","200","500"})` ef (beam width = L)
- `@Param({"true","false"})` twoPass
- `@BenchmarkMode(Throughput)`, `@OutputTimeUnit(SECONDS)` (= QPS)
- `@AuxCounters` for recall@10 — compute from SIFT ground-truth in `@Setup`

### 7b — `VamanaSearchBenchmark`

**Create** `vectors-bench/.../bench/VamanaSearchBenchmark.java`:
- Same structure as HNSW search benchmark

### 7c — `IvfSearchBenchmark` (only after Phase 4)

**Create** `vectors-bench/.../bench/IvfSearchBenchmark.java`:
- `@Param({"1","2","4","8","16"})` nprobe sweep

### 7d — `VectorDbBenchmark` (facade overhead)

**Create** `vectors-bench/.../bench/VectorDbBenchmark.java`:
- Wraps `VectorCollection.search()` — measures facade lock + metadata hydration overhead above raw index
- Compare against raw `HnswSearchBenchmark` at same ef to quantify wrapper cost

---

## Phase 8 — S3StorageBackend + LocalStack Integration Test

**Goal**: real S3 semantics for `DistributedVectorCollection` T3 tier.  
**Gate**: `P22 — DistributedVectorCollectionLocalStackIT` (`@Tag("integration")`, requires Docker).

### 8a — `S3StorageBackend`

**Create** `vectors-storage/.../backend/S3StorageBackend.java`:
- Implements `StorageBackend`
- AWS SDK v2 S3 client (already a test dependency in vectors-storage for LocalStack)
- `put(key, bytes)` → `PutObjectRequest`
- `get(key)` → `GetObjectRequest` with byte[] response
- `conditionalPut(key, bytes, expectedEtag)` → S3 `if-none-match: *` or `if-match: <etag>`
- `list(prefix)` → `ListObjectsV2Request` with prefix filter

**Modify** `vectors-storage/build.gradle.kts`:
- Move AWS SDK v2 S3 from `testImplementation` → `implementation` (if not already there)

### 8b — LocalStack integration test

**Create** `vectors-ivf/.../DistributedVectorCollectionLocalStackIT.java`:
- `@Testcontainers`, `@Tag("integration")`
- `@Container static LocalStackContainer localstack = new LocalStackContainer(...).withServices(S3)`
- Build `DistributedVectorCollection` with `S3StorageBackend(localstack.getEndpoint(), bucket, credentials)`
- Tests (reuse same scenario structure as `DistributedVectorCollectionIT`):
  - WAL replay restores committed vectors
  - T3 snapshots appear in S3 bucket
  - `conditionalPut` prevents double-commit (real S3 etag check)

**Add to** `vectors-ivf/build.gradle.kts`:
- `testImplementation("org.testcontainers:localstack:...")`
- `testImplementation("org.testcontainers:junit-jupiter:...")`
- `testImplementation(software.amazon.awssdk:s3:...)`

---

## Phase 9 — HyperDoor Cross-Tier Integration

**Goal**: `HyperDoor` address record drives eviction and load decisions in `TieredCluster`.  
**Gate**: `P23 — HyperDoorEvictionTest` (`@Tag("unit")`).

### 9a — `HyperDoor` → `TieredCluster` eviction signal

**Modify** `TieredCluster`:
- Add `HyperDoor hyperDoor()` — returns the cluster's tier address (node, shard, tier)
- `evictToTier(TierPolicy.Tier target, StorageBackend backend)` — evict T1 if target < T1;
  write T2 to `backend` if target < T2; ensure T3 is always present

### 9b — `TieredClusterRegistry`

**Create** `vectors-ivf/.../TieredClusterRegistry.java`:
- Maps `HyperDoor` → `TieredCluster`
- `lookup(HyperDoor)` → `Optional<TieredCluster>`
- Used by `DistributedVectorCollection` to support cross-shard cluster references (future distributed search)

### 9c — Gate test

**Create** `vectors-ivf/.../HyperDoorEvictionTest.java` (`@Tag("unit")`):
- `evictToT2_persistsToBackend_and_clearsT1`
- `evictToT3_persistsToT3_and_clearsT1andT2`
- `lookup_returnsClusterForKnownHyperDoor`
- `lookup_returnsEmptyForUnknownHyperDoor`

---

## P-Number Registry (TDD gate tests)

| P# | Class | Phase | Tag | Core assertion |
|---|---|---|---|---|
| P1–P10 | (committed) | Phase 1 | unit | — |
| P11 | `ClusterSplitterTest` | Phase 2a | unit | Quake cost model + bisect |
| P12 | `SubBuoyTreeTest` | Phase 2a | unit | hierarchical routing recall |
| P13 | `TierPolicyTest` | Phase 2b | unit | threshold promotion monotonicity |
| P14 | `TieredClusterTest` | Phase 2b | unit | T1 materialise, T3 round-trip |
| P15–P17 | `DistributedVectorCollectionTest` + IT | Phase 3 | unit + integration | WAL replay, conditional-put |
| **P18** | `VectorCollectionIvfFlatTest` | **Phase 4** | unit | IVF_FLAT recall ≥ 0.80, reopen, filter, delete |
| **P19** | `DistanceKernelBenchmark` | **Phase 5** | benchmark | Panama > scalar QPS on dim=128 |
| **P20** | `HnswIndexBuildBenchmark` + `VamanaIndexBuildBenchmark` | **Phase 6** | benchmark | completes without OOM |
| **P21** | `HnswSearchBenchmark` + `VamanaSearchBenchmark` | **Phase 7** | benchmark | recall@10 ≥ 0.95 on SIFT Small (via AuxCounters) |
| **P22** | `DistributedVectorCollectionLocalStackIT` | **Phase 8** | integration | real S3 etag, WAL replay |
| **P23** | `HyperDoorEvictionTest` | **Phase 9** | unit | evict T1→T3, registry lookup |

---

## Deferred / Out of Scope

The following are captured in `ideas-to-explore.md` and the deep-research report but are
explicitly **not** part of this plan — they require new research before committing to an approach:

- **Graph segment merge / compaction** — the `compact()` path in `VectorCollectionImpl` exists
  but the merge strategy (incremental Vamana reinsertion vs. full rebuild) is unresolved.
- **RaBitQ-based BuoyIndex T0 tier** — replacing the centroid routing with 1-bit BuoyIndex
  for sub-linear routing cost; requires RaBitQ quantized cluster routing research.
- **Distributed gossip / shard coordination** — multi-JVM `DistributedVectorCollection`;
  requires protocol design (Raft / CRDTs).
- **GPU / OpenCL off-load** — out of scope (project is pure-Java by design).
- **SIMD prefix-sum for BQ** — Lucene-style bit-level distance with prefix accumulation.

---

## Quick Reference: Dependency Table

| Phase | Depends on | Blocks |
|---|---|---|
| 4 (IVF_FLAT) | vectors-ivf complete ✅ | Phase 6c, 7c |
| 5 (Tier 1 JMH) | vectors-core complete ✅ | nothing |
| 6 (Tier 2 JMH) | Phase 5 pattern; Phase 4 for 6c | Phase 7 |
| 7 (Tier 3 JMH) | Phase 6; Phase 4 for 7c | nothing |
| 8 (S3 backend) | vectors-storage complete ✅ | Phase 9 |
| 9 (HyperDoor) | vectors-ivf complete ✅; Phase 8 | nothing |
