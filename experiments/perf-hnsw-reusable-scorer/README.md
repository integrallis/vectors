# perf(hnsw): eliminate per-search default-scorer allocation

## Hypothesis
`HnswSearcher.defaultScorer(float[])` allocates a new anonymous `NodeScorer` instance on every
search call. For the default full-precision path (the common case), this is one heap object per
query — small but unnecessary. A stateful scorer whose query reference is swapped in-place
eliminates the allocation with no algorithmic change.

## Decision rule
If all HNSW tests pass and JMH shows no regression, accept.

## Kill criterion
If any searcher test fails because a returned scorer is unexpectedly shared, reject.

## Implementation
Add a `MutableNodeScorer` internal interface with `setQuery(float[])`. Create four reusable
implementations matching the existing paths:
- `FusedScorer` — fused-GEMV full-precision path
- `SimpleScorer` — `sharesReturnBuffer()` fallback
- `SegmentScorer` — zero-copy MemorySegment path
- `CosineSegmentScorer` — cosine with segment-matrix batching

The constructor pre-creates the appropriate reusable scorer. `prepareDefaultScorer(float[])`
updates its query reference and returns the same instance every call.

## Files changed
- `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/HnswSearcher.java`

## Testing
- `vectors-hnsw:test` — 155 tests, ALL PASS
- `vectors-db:test` (VectorDbTest, FlatCollectionBatchTest) — PASS
- `spotlessCheck` — PASS
- `spotbugsMain` — PASS
- JMH compile — PASS

## Note on measurement
Local dev-machine JMH runs are too noisy (error bars exceed the score) to yield reliable
before/after comparison. A quiet-VPS run of `AnnSearchBenchmark.searchHnsw` and
`VectorDbBenchmark.searchRawHnsw` is needed for a clean delta.
