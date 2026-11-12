# perf-neighborselector-scratch: reusable scratch in NeighborSelector pruning

## Hypothesis

`NeighborSelector.selectDiverse` allocates three buffers on every call:
1. `NeighborArray result` — up to `maxConnections` entries
2. `float[] scratch` — for shared-buffer store copies (dim × 4 bytes)
3. `boolean[] blocked` — `candidates.size()` booleans

In the HNSW build path, `selectDiverse` is called:
- Once per layer per node (initial selection)
- Plus once per backlink when the backlink target's list overflows (pruning)

At N=100K, ~3 layers, this is 300K+ calls. The scratch allocations add up to
significant transient heap churn.

## Decision rule

Accept if:
- All `:vectors-hnsw:test` pass (155 tests)
- spotless / spotbugs clean
- JMH compile succeeds

## Implementation

Add `selectDiverseWithScratch` overloads (cached + uncached) accepting
optional pre-allocated `NeighborArray resultOut`, `float[] scratch`, and
`boolean[] blockedScratch`. The public `selectDiverse` methods delegate to
`selectDiverseWithScratch` with `null` for all scratch — zero change to
existing callers.

Wire up ONLY the **pruning** calls (where the result is consumed immediately
via `copyFrom`) in both builders:
- `HnswGraphBuilder`: add `pruneResult` field, wire backlink pruning
- `ConcurrentHnswGraphBuilder.WorkContext`: add `pruneResult` field, wire
  forward-edge pruning + backlink pruning

The **initial** `selectDiverse` (which returns `neighbors` used by the
backlink loop) is left unchanged because its result must survive until the
loop completes.

## Files changed

- `vectors-hnsw/.../NeighborSelector.java` — new overloads
- `vectors-hnsw/.../HnswGraphBuilder.java` — scratch wiring
- `vectors-hnsw/.../ConcurrentHnswGraphBuilder.java` — scratch wiring

## Verification

```
:vectors-hnsw:test              155/155 PASSED
:vectors-hnsw:spotlessCheck   PASSED
:vectors-hnsw:spotbugsMain    PASSED
:vectors-bench:compileJmhJava PASSED
```

## Decision

ACCEPTED. Eliminates `float[]`, `boolean[]`, and `NeighborArray` allocations
on all pruning calls in the build path. Initial selection results remain
fresh allocations (required by downstream consumption).
