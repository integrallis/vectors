# perf-rescore-heap-sizing: size rescore heap for typical two-pass workloads

## Hypothesis

`HnswSearcher` and `VamanaSearcher` both allocate `rescoreHeap` with capacity 64.
During two-pass quantized search, `rescore()` ingests `coarseK = max(k, k *
overQueryFactor)` candidates. For k=100 and overQueryFactor=2.0, coarseK=200.

Inserting 200 entries into a heap of capacity 64 triggers geometric growth:
64 → 96 → 144 → 216 (3 reallocations + array copies on every rescore call).

Sizing the heap to 256 eliminates these growth events for typical k ≤ 256
while adding only ~2KB per searcher (the dominant memory is already the
visitedTag array).

## Decision rule

Accept if:
- `:vectors-hnsw:test` and `:vectors-vamana:test` both pass
- spotless / spotbugs clean
- JMH compile succeeds

## Implementation

Change `rescoreHeap` initial capacity from 64 to 256 in both searchers.

## Files changed

- `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/HnswSearcher.java`
- `vectors-vamana/src/main/java/com/integrallis/vectors/vamana/VamanaSearcher.java`

## Verification

```
:vectors-hnsw:spotlessCheck      PASSED
:vectors-hnsw:spotbugsMain     PASSED
:vectors-vamana:spotlessCheck  PASSED
:vectors-vamana:spotbugsMain   PASSED
:vectors-hnsw:test              14/14 PASSED
:vectors-vamana:test            14/14 PASSED
:vectors-bench:compileJmhJava   PASSED
```

## Decision

ACCEPTED. Removes mid-rescore heap growth for the common two-pass regime
without measurably increasing per-searcher memory footprint.
