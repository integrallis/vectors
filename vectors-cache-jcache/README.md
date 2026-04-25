# vectors-cache-jcache

JSR-107 (JCache) adapter for the `VectorCache` SPI. Wraps any JCache-compliant backend (Ehcache, Hazelcast, Infinispan, Caffeine-JCache) as a `VectorCache`.

## Responsibility

- `JCacheVectorCache` adapts a `javax.cache.Cache` to the `VectorCache` interface
- Tracks hits and misses via `LongAdder` (does not rely on provider statistics)

## Key Types

- `JCacheVectorCache` — adapter class

## Dependencies

- `vectors-cache` — VectorCache SPI
- JCache API 1.1.1 (`javax.cache:cache-api`)
