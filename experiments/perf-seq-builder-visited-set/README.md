# perf-seq-builder-visited-set: replace BitSet with generation-stamped int[] in sequential HNSW builder

## Hypothesis

`HnswGraphBuilder` (single-threaded) uses `java.util.BitSet` for the visited set
during beam search. `BitSet.clear()` is called once per insertion (which means
once per layer per node). At N=100K and ~3 layers, that's 300K `clear()` calls.
Each `clear()` zeroes `N/64` longs — at N=100K that's 1563 longs = 12.5KB per
call. Total memory writes: ~300K × 12.5KB ≈ 3.75 GB of zeroing during a full
build.

Replacing `BitSet` with a generation-stamped `int[]` eliminates this O(N) per-
search cost.

Additionally, `searchLayer()` calls `NeighborArray.drainResults(results)` which
allocates a fresh `NeighborArray` on every call. A reusable `NeighborArray` sized
to `efConstruction` removes this per-search allocation.

## Decision rule

Accept if:
- `:vectors-hnsw:test` passes (all 14 tests)
- spotless / spotbugs clean
- JMH compile succeeds

## Implementation

- Replace `final BitSet visited` with `final int[] visitedTag` + `int visitedGen`
- Add `nextVisitedGeneration()` helper
- Swap `visited.clear/get/set` calls for `visitedTag` array access
- Add `final NeighborArray searchResults` sized to `efConstruction`
- In `searchLayer()`: `searchResults.clear(); searchResults.drainResultsFrom(results);`
  instead of `NeighborArray.drainResults(results)`

## Files changed

- `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/HnswGraphBuilder.java`

## Verification

```
:vectors-hnsw:spotlessCheck      PASSED
:vectors-hnsw:spotbugsMain      PASSED
:vectors-hnsw:test                14/14 PASSED
:vectors-bench:compileJmhJava     PASSED
```

## Decision

ACCEPTED. Brings the sequential build path into parity with the concurrent
build path optimizations already landed. Zero functional change.
