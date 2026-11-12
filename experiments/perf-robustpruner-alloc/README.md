# perf-robustpruner-alloc: reusable scratch in RobustPruner

## Hypothesis

`RobustPruner.robustPrune()` is the core diversity-pruning routine in the
Vamana build path. It is called **once per node per pass** (≈ N × passes
times). On every call it allocates two scratch buffers:

1. `boolean[] eligible` — sized to `candidates.size()` (up to
   `searchListSize + maxDegree + 1`, typically ~200).
2. `float[] candidateScratch` — sized to `vectors.dimension()` (up to 768
   or 1536 bytes), **only when the backing store is a shared-buffer mmap
   implementation**.

At N=100K, dim=768, 2 passes, on a shared-buffer store:
- `eligible`: 200K allocations × ~200 B  = ~40 MB heap churn
- `candidateScratch`: 200K allocations × 3 KB = **~600 MB heap churn**

These are pure waste — the buffers are transient to a single `robustPrune`
call and never escape.

## Decision rule

Accept if:
- All `:vectors-vamana:test` pass (14/14)
- spotless / spotbugs clean
- JMH compile succeeds

## Implementation

Add a 9-argument overload to `RobustPruner.robustPrune` that accepts optional
`float[] candidateScratch` and `boolean[] eligibleScratch`. The original
7-argument method delegates to the 9-argument one with `null` for both buffers,
preserving behaviour for all existing callers (including 10+ tests).

Update the two hot build callers:
- `VamanaGraphBuilder.buildPass` (sequential)
- `ConcurrentVamanaGraphBuilder.processNodeConcurrent` (concurrent)

Both now create the scratch once per pass / per thread and pass it to every
`robustPrune` invocation.

## Files changed

- `vectors-vamana/src/main/java/com/integrallis/vectors/vamana/RobustPruner.java`
- `vectors-vamana/src/main/java/com/integrallis/vectors/vamana/VamanaGraphBuilder.java`
- `vectors-vamana/src/main/java/com/integrallis/vectors/vamana/ConcurrentVamanaGraphBuilder.java`

## Verification

```
:vectors-vamana:spotlessCheck      PASSED
:vectors-vamana:spotbugsMain     PASSED
:vectors-vamana:test               14/14 PASSED
:vectors-bench:compileJmhJava      PASSED
```

## Decision

ACCEPTED. Eliminates ~640MB of transient allocation per 100K-node build on
shared-buffer stores. Zero functional change; all existing callers (tests and
sequential builder) are unaffected except for the two hot paths being fixed.
