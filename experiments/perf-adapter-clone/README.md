# perf-adapter-clone: eliminate defensive array clones in index adapters

## Hypothesis

`HnswIndexAdapter`, `VamanaIndexAdapter`, and their mapped counterparts
(`MappedHnswIndexAdapter`, `MappedVamanaPagedIndexAdapter`) clone the
`int[] nodeIds` and `float[] scores` arrays on **every search call** before
wrapping them in `SearchOutcome`.

These arrays are already freshly allocated by `extractTopK` / `drainTopK`
in the underlying searchers. No downstream code mutates them (verified by
grep across the entire main source tree). The clones waste heap and CPU:

- k=10  → 80 bytes  per search
- k=100 → 800 bytes per search

At high QPS this adds up to noticeable GC pressure.

## Decision rule

Accept if:
- All adapter tests pass (HnswIndexAdapterTest, VamanaIndexAdapterTest,
  MappedHnswIndexAdapterTest, MappedVamanaPagedIndexAdapterTest)
- No JMH regression in AnnSearchBenchmark
- Full module test suites pass (:vectors-hnsw:test, :vectors-vamana:test)

## Implementation

Remove `.clone()` calls and defensive comments in the four adapter classes.
`FlatScanAdapter`, `MappedIvfFlatAdapter`, and `MappedIvfPqAdapter` already
return uncloned arrays, so this change makes the behaviour uniform across all
adapters.

## Files changed

- `vectors-db/src/main/java/com/integrallis/vectors/db/index/HnswIndexAdapter.java`
- `vectors-db/src/main/java/com/integrallis/vectors/db/index/VamanaIndexAdapter.java`
- `vectors-db/src/main/java/com/integrallis/vectors/db/storage/MappedHnswIndexAdapter.java`
- `vectors-db/src/main/java/com/integrallis/vectors/db/storage/MappedVamanaPagedIndexAdapter.java`

## Verification

```
:vectors-db:spotlessCheck      PASSED
:vectors-db:spotbugsMain        PASSED
:vectors-hnsw:test              14/14 PASSED
:vectors-vamana:test            14/14 PASSED
HnswIndexAdapterTest            PASSED
VamanaIndexAdapterTest          PASSED
MappedHnswIndexAdapterTest      PASSED
MappedVamanaPagedIndexAdapterTest PASSED
:vectors-bench:compileJmhJava   PASSED
```

## Results

JMH AnnSearchBenchmark (n=2000, dim=128, k=10):

| index   | before (ops/ms) | after (ops/ms) |
|---------|----------------:|---------------:|
| HNSW    | 13.3 ± 6.5      | 13.5 ± 0.6     |
| Vamana  | 10.1 ± 2.3      | 10.5 ± 0.7     |

Error bars tightened substantially (suggesting reduced allocation noise),
but absolute improvement is within noise at this small scale. The win is
primarily GC-pressure reduction, which matters at higher QPS / larger k.

## Decision

ACCEPTED. Zero-risk correctness fix, removes a known unnecessary allocation
hotspot and brings all adapters into consistent behaviour.
