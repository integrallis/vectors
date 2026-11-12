# perf-gen-prefilter-flag: cache ACORN pre-filter capability in Generation

## Hypothesis

`VectorCollectionImpl.search()` evaluates a 4-way `instanceof` chain on every
call to decide whether the active SPI supports ACORN pre-filter navigation:

```java
boolean supportsPreFilter =
    gen.spi instanceof HnswIndexAdapter
        || gen.spi instanceof MappedHnswIndexAdapter
        || gen.spi instanceof VamanaIndexAdapter
        || gen.spi instanceof MappedVamanaPagedIndexAdapter;
```

`instanceof` on concrete classes is a simple class-pointer comparison, but 4
back-to-back comparisons is still 4 branches per search. For a given
collection the SPI type is invariant across its lifetime, so the result is
constant. Caching it in `Generation` eliminates the repeated dispatch.

## Decision rule

Accept if:
- Adapter tests pass (HNSW + Vamana, mapped and in-memory)
- `:vectors-db:spotlessCheck` and `:spotbugsMain` pass
- JMH compile succeeds

## Implementation

Add `final boolean supportsPreFilter` to `VectorCollectionImpl.Generation`,
computed once in the constructor from the same 4-way `instanceof` chain.
Replace the inline `instanceof` chain in `search()` with `gen.supportsPreFilter`.

## Files changed

- `vectors-db/src/main/java/com/integrallis/vectors/db/VectorCollectionImpl.java`

## Verification

```
:vectors-db:spotlessCheck      PASSED
:vectors-db:spotbugsMain     PASSED
Adapter tests (HNSW + Vamana)  PASSED
:vectors-bench:compileJmhJava  PASSED
```

## Decision

ACCEPTED. One-time cost in constructor, zero functional change, removes a
repeated branch chain from the hot search path.
