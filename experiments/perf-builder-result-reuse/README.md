# perf-builder-result-reuse: reusable search-result arrays in concurrent builders

## Hypothesis

`ConcurrentHnswGraphBuilder.searchLayerConcurrent()` allocates a fresh
`NeighborArray` on every call to drain the result min-heap. For a 100K-node build
with ~3 layers per node, that's ~300K `NeighborArray` allocations. At ef=200,
each `NeighborArray` carries `int[200]` + `float[200]` = 1.6KB — roughly 480MB
of short-lived heap churn per build.

`ConcurrentVamanaGraphBuilder.searchConcurrent()` similarly allocates fresh
`int[]` and `float[]` arrays on every search call (once per node). At
searchListSize=128, that's ~1KB × 100K = 100MB of heap churn.

Adding a reusable drain target to `WorkContext` eliminates all of these
per-search result-array allocations.

## Decision rule

Accept if:
- `:vectors-hnsw:test` and `:vectors-vamana:test` both pass
- spotless / spotbugs clean
- JMH compile succeeds
- No `ConcurrentHnswGraphBuilder.drainResults` callers break (tests exercise it)

## Implementation

### HNSW builder (`ConcurrentHnswGraphBuilder`)
- Add `final NeighborArray searchResults` to `WorkContext`, sized to `ef`
- In `searchLayerConcurrent`: clear + `drainResultsFrom(ctx.results)` into
  `ctx.searchResults`, return it directly
- Keep `drainResults(NodeQueue)` as a public compatibility shim (used by
  `ResultDrainTest`)

### Vamana builder (`ConcurrentVamanaGraphBuilder`)
- Add `final int[] searchNodeIds` and `final float[] searchScores` to
  `WorkContext`, both sized to `searchListSize`
- In `searchConcurrent`: drain into these scratch arrays instead of allocating
  new ones

## Files changed

- `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/ConcurrentHnswGraphBuilder.java`
- `vectors-vamana/src/main/java/com/integrallis/vectors/vamana/ConcurrentVamanaGraphBuilder.java`

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

ACCEPTED. Eliminates a known per-search allocation hotspot in the concurrent
build path. Zero functional change — only scratch buffer reuse.
