# perf(db): eliminate per-search query float[] clone on cosine-normalized collections

## Hypothesis
When a collection has `normalizeForCosine=true`, `VectorCollectionImpl.search()` clones the caller's query float[] before normalizing it. For a 768-dim query this allocates ~3KB per search call. At high QPS this is a steady young-gen allocation stream.

A per-thread scratch buffer can hold the normalized copy safely: the normalized query is consumed synchronously by the SPI search methods and never escapes the search method.

## Decision rule
If all search-path tests (QvCache, multi-start, flat batch, HNSW, Vamana) continue to pass, accept.

## Kill criterion
If any test fails because a returned result unexpectedly references the normalized query array (e.g., the SearchResult or cache key stores the float[] reference), reject.

## Implementation
- Add `ThreadLocal<float[]> QUERY_NORMALIZE_SCRATCH`
- `search()`: when `normalizeForCosine` is true, copy request.query() into the scratch, normalize in-place, and pass the scratch to the cache + SPI.
- No API change.
