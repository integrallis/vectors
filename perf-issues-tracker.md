# Performance Enhancement Issues — Ready to File

This tracker consolidates all performance branches from the Nov 2026 session.
Each row links to its branch and contains a ready-to-paste GitHub issue body.

## Issue #1 — Vamana visited set
**Branch:** `perf/vamana-visited-set`
**Files:** `vectors-vamana/src/.../VamanaSearcher.java`
**Estimated impact:** +10.4% search QPS

```
## Summary
Replace BitSet.clear() (O(N) per query) with a generation-stamped int[] visited set
(O(1) per query) in VamanaSearcher, matching the HnswSearcher optimization.

## Before
BitSet.clear() zeroes N/64 longs on every search call. At N=100K this is ~1563
long writes × queries/sec.

## After
Increment a generation counter; a node is "visited" iff visitedTag[id] == visitedGen.
No array zeroing. On int wraparound, fill once.

## Benchmark
JMH microbenchmark on perf-microbench dataset (n=10,000, dim=128):
- Before: 1,482 ops/ms
- After: 1,636 ops/ms (+10.4%)

## Verification
- :vectors-vamana:test 14/14 PASSED
- :vectors-vamana:spotlessCheck PASSED
- :vectors-vamana:spotbugsMain PASSED
```

## Issue #2 — Searcher heap sizing
**Branch:** `perf/searcher-heap-sizing`
**Files:** `vectors-hnsw/.../HnswSearcher.java`, `vectors-vamana/.../VamanaSearcher.java`
**Estimated impact:** +29% NodeQueue throughput

```
## Summary
Size candidate/results scratch heaps by graph degree instead of hard-coded 256.
HnswSearcher: candidates=max(512, maxConnections0*4), results=max(384, maxConnections0*2).
VamanaSearcher: candidates=max(256, maxDegree*4), results=max(128, maxDegree*2).

## Before
Fixed initial capacity 256 caused geometric growth for efSearch ≥ 500.

## After
Adaptive sizing based on actual graph degree eliminates mid-search heap growth.

## Benchmark
NodeQueueGrowthMicrobench (1K ops × 1K iterations, M=16):
- Before: 8,519 ops/ms
- After: 10,987 ops/ms (+29%)

## Verification
- :vectors-hnsw:test 14/14 PASSED
- :vectors-vamana:test 14/14 PASSED
- spotless, spotbugs, JMH compile PASSED
```

## Issue #3 — Vamana prefetch hook
**Branch:** `perf/vamana-prefetch`
**Files:** `vectors-vamana/.../VamanaSearcher.java`, `vectors-vamana/.../VamanaIndex.java`
**Estimated impact:** Feature parity (SSD prefetch), zero warm-path overhead

```
## Summary
HnswSearcher exposes setPrefetchHook(IntConsumer) for AsyncVectorPrefetcher
overlap. VamanaSearcher had no equivalent.

## After
Add prefetchHook field + setter. Call before bulkScore in search()/searchFiltered().
Add VamanaIndex.searchWithPrefetch() mirroring HnswIndex.

## Benchmark
Zero overhead when hook == null (branch predicted away).

## Verification
- :vectors-vamana:test 14/14 PASSED
```

## Issue #4 — pickSeeds allocation
**Branch:** `perf/pickseeds-alloc`
**Files:** `vectors-hnsw/.../HnswSearcher.java`
**Estimated impact:** Eliminates 2× per-search allocation on multi-start path

```
## Summary
pickSeeds() allocates new int[]{entry} for single-layer graphs and new int[count]
for every multi-start seed selection.

## After
Use existing singleEntry field + add bounded seedsScratch (size 32).

## Verification
- :vectors-hnsw:test 14/14 PASSED
```

## Issue #5 — NeighborArray linear scan
**Branch:** `perf/neighbor-insert`
**Files:** `vectors-hnsw/.../NeighborArray.java`, `vectors-vamana/.../NeighborArray.java`
**Estimated impact:** +54–70% insert throughput for small arrays

```
## Summary
NeighborArray.insert() used binary search + System.arraycopy for all sizes.
For capacity ≤ 64, JNI overhead of tiny arraycopies dominates.

## After
Backwards linear scan + manual shift for arrays ≤ 64 (LINEAR_SCAN_THRESHOLD).

## Benchmark (100K inserts × 1K iterations):
- cap=16: 18,900 → 29,205 ops/ms (+54.5%)
- cap=32: 14,277 → 24,275 ops/ms (+69.9%)
- cap=64: 10,521 → 16,854 ops/ms (+60.2%)

## Verification
- :vectors-hnsw:test 14/14 PASSED
- :vectors-vamana:test 14/14 PASSED
```

## Issue #6 — Adapter defensive clones
**Branch:** `perf/adapter-clone`
**Files:** 4 SPI adapters
**Estimated impact:** -80–800 bytes allocation per search

```
## Summary
HnswIndexAdapter, VamanaIndexAdapter, MappedHnswIndexAdapter, and
MappedVamanaPagedIndexAdapter all called .clone() on fresh int[]/float[]
arrays before wrapping in SearchOutcome.

## After
Remove .clone() calls. FlatScanAdapter and IVF adapters already returned
uncloned arrays; this makes behaviour uniform.

## Verification
- Adapter tests PASSED
- :vectors-hnsw:test 14/14 PASSED
- :vectors-vamana:test 14/14 PASSED
```

## Issue #7 — Concurrent builder visited set
**Branch:** `perf/builder-visited-set`
**Files:** `vectors-hnsw/.../ConcurrentHnswGraphBuilder.java`, `vectors-vamana/.../ConcurrentVamanaGraphBuilder.java`
**Estimated impact:** 3.8–41× faster visited reset per search

```
## Summary
Concurrent builders used BitSet for per-node beam-searchvisited tracking.
BitSet.clear() is O(N/64) per search.

## After
Generation-stamped int[] pattern (same as searchers).

## Microbenchmark (50K iterations):
- n=1K:  3.8× speedup
- n=5K:  6.7× speedup
- n=20K: 41.8× speedup
- n=100K: 3.9× speedup

## Verification
- :vectors-hnsw:test 14/14 PASSED
- :vectors-vamana:test 14/14 PASSED
```

## Issue #8 — Concurrent builder result reuse
**Branch:** `perf/builder-result-reuse`
**Files:** `vectors-hnsw/.../ConcurrentHnswGraphBuilder.java`, `vectors-vamana/.../ConcurrentVamanaGraphBuilder.java`
**Estimated impact:** ~480MB+ heap churn eliminated per 100K-node build

```
## Summary
ConcurrentHnswGraphBuilder allocated fresh NeighborArray per searchLayerConcurrent.
ConcurrentVamanaGraphBuilder allocated fresh int[]/float[] per searchConcurrent.

## After
Add reusable drain targets to WorkContext.

## Verification
- :vectors-hnsw:test 14/14 PASSED
- :vectors-vamana:test 14/14 PASSED
```

## Issue #9 — Sequential builder visited set
**Branch:** `perf/seq-builder-visited-set`
**Files:** `vectors-hnsw/.../HnswGraphBuilder.java`
**Estimated impact:** ~3.75GB memory writes eliminated per 100K-node build

```
## Summary
HnswGraphBuilder (single-threaded) also used BitSet.clear() once per layer
per node. Replace with generation-stamped int[]. Also add reusable
NeighborArray for searchLayer result drain.

## Verification
- :vectors-hnsw:test 14/14 PASSED
```

## Issue #10 — Generation pre-filter flag
**Branch:** `perf/gen-prefilter-flag`
**Files:** `vectors-db/.../VectorCollectionImpl.java`
**Estimated impact:** Removes 4-instanceof branch chain per search

```
## Summary
VectorCollectionImpl.search() evaluated a 4-way instanceof chain on every
call. The SPI type is invariant.

## After
Cache supportsPreFilter boolean in Generation constructor.

## Verification
- Adapter tests PASSED
- :vectors-db:spotlessCheck + spotbugsMain PASSED
```

## Issue #11 — Rescore heap sizing
**Branch:** `perf/rescore-heap-sizing`
**Files:** `vectors-hnsw/.../HnswSearcher.java`, `vectors-vamana/.../VamanaSearcher.java`
**Estimated impact:** Eliminates 3 reallocations per two-pass search

```
## Summary
rescoreHeap capacity 64 causes growth at coarseK=200 (k=100, OQF=2.0):
64 → 96 → 144 → 216 (3 growth events).

## After
Resize to 256. Eliminates growth for k ≤ 256, adds ~2KB per searcher.

## Verification
- :vectors-hnsw:test 14/14 PASSED
- :vectors-vamana:test 14/14 PASSED
```

## Issue #12 — RobustPruner scratch reuse
**Branch:** `perf/robustpruner-alloc`
**Files:** `vectors-vamana/.../RobustPruner.java`, `.../VamanaGraphBuilder.java`, `.../ConcurrentVamanaGraphBuilder.java`
**Estimated impact:** ~640MB transient allocation eliminated per 100K build

```
## Summary
robustPrune allocated boolean[] eligible + float[] candidateScratch on
every call. At N=100K, 2 passes, shared-buffer store, dim=768:
- eligible: ~200K × 200 B = ~40 MB
- candidateScratch: ~200K × 3 KB = ~600 MB

## After
Add 9-arg overload with reusable scratch. Sequential + concurrent builders
instantiate scratch once per pass / per thread.

## Verification
- :vectors-vamana:test 14/14 PASSED
```
