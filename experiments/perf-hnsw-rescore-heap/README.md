# perf(hnsw): size rescore heap to 256 to avoid mid-rescore growth

## Hypothesis
`HnswSearcher.rescoreHeap` is currently 64, same as the old `VamanaSearcher` value. During two-pass search with quantized coarse filtering and k as low as 100, the rescore pass may rank more than 64 candidates (coarseK = k * overQueryFactor, default 2.0). When the heap saturates it triggers a grow-and-copy in `NodeQueue.ensureCapacity`, adding a mid-search allocation.

## Decision rule
If Benchmark and AllocationProfiler show zero growth events in a multi-k two-pass search, accept.

## Kill criterion
If total allocation increases because 256 is too large for the typical search, reject.

## Implementation
Size the rescore heap to 256, matching `VamanaSearcher`.
