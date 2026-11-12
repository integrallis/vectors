# perf(cache): zero-allocation QvCache lookup on cache hit

## Hypothesis
VectorCollectionImpl.search() calls queryCache.buildKey() on every search, which calls QvCache.quantize(). quantize allocates a new byte[dim] on every call. For a 768-dim query that is 768 bytes per search. At 10K QPS on a 50% hit-rate cache, half of those allocations are wasted — the key is looked up, found, and then the key (and its byte[]) become garbage immediately.

## Decision rule
If QvCache tests still pass with correct hit/miss behavior, accept.

## Kill criterion
If the cache returns stale results because the scratch buffer is overwritten between get() and put(), reject.

## Implementation
1. Add QvCache.get(float[], int, Filter, float, boolean, boolean) overload that uses a ThreadLocal<byte[]> scratch for quantization.
2. On hit: returns the cached SearchResult with zero allocation.
3. On miss: VectorCollectionImpl falls back to buildKey() to create a permanent key for put().
4. Add quantize(float[], byte[]) overload for in-place quantization without allocation.
