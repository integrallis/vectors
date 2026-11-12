# perf(hnsw): cap multi-start worker creation at availableProcessors

## Hypothesis
searchMultiStart currently spawns one virtual-thread task per seed, and each task creates a brand-new HnswSearcher (including a ~4MB visitedTag array for a 1M-node graph). For 32 seeds, this allocates ~32 × 4.5 MB = 144 MB of transient heap per call.

CPU-bound work does not benefit from more concurrent workers than physical cores. Capping worker creation at availableProcessors and having each worker process a chunk of seeds sequentially eliminates the excess allocation without losing parallelism.

## Decision rule
If multi-start search tests and recall benchmarks pass with identical results, accept.

## Kill criterion
If deduplication or early-termination logic changes because workers now process multiple seeds, reject.

## Implementation
- Extract beamSearch0 helper with a boolean reuseResult parameter
- beamSearch reuses reusableResult (existing behavior)
- beamSearchFreshResult allocates a new NeighborArray via NeighborArray.drainResults (safe for sequential reuse of the same worker)
- runParallelBeams chunks seeds across at most availableProcessors workers
- Each worker processes its chunk sequentially, returning a List<NeighborArray>
