# vectors-cache

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/vectors/main/vectors-cache/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Caching SPI for exact-match and similarity-based lookups. Provides `VectorCache` for key-value caching and `SemanticCache` for vector-similarity retrieval, with a Caffeine default backend.

## Responsibility

- `VectorCache<K, V>` — exact-match cache SPI with atomic `getOrCompute()` hot path
- `SemanticCache<V>` — similarity-search cache interface (threshold-based k-NN)
- `CaffeineVectorCache` — Caffeine-backed implementation with builder pattern
- `NoOpVectorCache` — no-op sink for benchmarking and wiring
- `CacheStats` — hit/miss/eviction/size counters with `hitRate()` helper

## Key Types

- `VectorCache` — core SPI interface
- `SemanticCache` — similarity-based cache interface
- `CaffeineVectorCache` — default implementation
- `CacheStats` — statistics record

## Dependencies

- Caffeine 3.1.8
