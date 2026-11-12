# perf(vamana): generation-stamped int[] visited set in concurrent builder

## Hypothesis
In the path we already optimized the non-concurrent VamanaSearcher (from BitSet.clear() to a generation-stamped int[]), the remaining BitSet allocation is in ConcurrentVamanaGraphBuilder's per-thread WorkContext, which allocates `new BitSet(maxNodes)` for every pass, and then calls `.clear()` at the start of every beam search. Both are cheap in absolute terms, but in a concurrent build with large graphs they scale linearly with node count: BitSet.clear() does O(N/8) byte-clears on every search call, and the BitSet constructor itself allocates an array of that size.

## Decision rule
Compile, run existing Vamana tests. If green, accept.

## Kill criterion
If any concurrent-build test fails because the int[] isn't being reset correctly, reject.

## Implementation
- Replace BitSet with int[] visitedTag + int visitedGen in WorkContext
- Use same pattern as HnswSearcher/ConcurrentHnswGraphBuilder
- Clear via generation bump + wraparound reset
