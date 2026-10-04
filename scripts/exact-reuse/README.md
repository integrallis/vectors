# Exact reuse experiments

Baseline: merged main `212ad282d719a985706b8024ad310c73e5e85430`.

This round separates three hypotheses:

1. Heap COSINE batches can compute each stored row's squared norm once per scan,
   then reuse it across queries. Query norms are already reused on main. No norms
   persist across calls; mutations remain visible. The provider automatically uses
   its original arithmetic for short vectors and scalar execution.
2. **Rejected:** concurrent HNSW diversity pruning can use the existing exact symmetric pair
   cache independently in each worker. Full ordinal keys distinguish collisions;
   each cache belongs to one vector source, metric and build/append operation.
   Cache storage is 192 KiB plus array headers per active worker. M=16,
   efConstruction=200 and query ef remain unchanged. It reduced a small fixed
   probe's reads but regressed construction time; it is absent from the final runtime.
3. Concurrent HNSW beam results can use the already-qualified exact heap drain,
   removing two temporary arrays and quadratic duplicate scans. Tie order remains
   identical. This is measured independently of both flat scoring and the rejected cache.

Neither experiment introduces a runtime option. Separate frozen jar directories
provide the experimental ablation; applications use the qualified defaults.

## Gates fixed before measurements

- Red tests precede implementation. Flat scores retain exact warmed float bits,
  IDs and mutation behavior; scalar/128-bit SIMD are checked independently.
- Flat batch median time must improve by at least 5% on both GLOVE COSINE and the
  retained 512-dimensional COSINE collection. Query counts 1, 2, 4, 16 and 64,
  repeated individual searches and Fashion-MNIST EUCLIDEAN expose controls.
- Concurrent construction median time must improve by at least 5% on both
  published corpora. Each paired recall@10 delta at ef=32/128/512 must be at least
  -0.005; this is the existing gate, not a new allowance. Report all deltas.
- A fixed single-worker schedule must produce exactly the same encoded graph and
  search result digest. Multiple worker schedules are inherently nondeterministic;
  compare several paired seeds and disclose both recall and search latency.
- Frozen jars, SHA-256 input/harness hashes, exact commands and raw samples are
  retained. Only one measured process runs on the dedicated VPS at a time.
- CI must pass before promotion. Failed candidates remain experiments, without
  introducing opt-ins or weakening gates.

## Reproduction

Use Java 25 with the Vector API on one otherwise idle machine. Build each revision
before measuring. The HNSW arm is baseline jars with only `vectors-hnsw` replaced;
this prevents the flat change from affecting its attribution.

```sh
JAVA_HOME=/path/to/jdk25 ./gradlew -I scripts/exact-reuse/freeze.gradle \
  :vectors-db:freezePerf -PfreezeDir=/absolute/path/to/frozen-jars \
  --no-daemon --max-workers=4

python3 scripts/exact-reuse/run-flat.py BASELINE_JARS FLAT_JARS \
  PUBLISHED_INPUTS SAVED_COLLECTION OUTPUT \
  --baseline-sha BASELINE_SHA --candidate-sha FLAT_SHA

python3 scripts/exact-reuse/run-construction.py BASELINE_JARS HNSW_JARS \
  PUBLISHED_INPUTS OUTPUT --rows 20000 --seeds 17,42,91 --threads 1,8 \
  --baseline-sha BASELINE_SHA --candidate-sha HNSW_SHA
```

`PUBLISHED_INPUTS` contains the four train/test fbin files listed in the evidence
provenance. An fbin begins with little-endian int32 row and dimension counts,
followed by row-major float32 values. Exact neighbors are recomputed over each
selected construction prefix using the baseline, never copied from ground truth
for the full dataset. The saved collection has `manifest.bin` and `vectors.bin`;
its deterministic perturbed source queries are throughput/parity probes, **not
held-out recall evidence**. GLOVE and Fashion use 256 official held-out queries.

For the norm decomposition rationale and limitations, see the primary
[Faiss implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes).
Reusing a norm does not implement its BLAS distance kernel. Likewise, bounded
streaming pair-score reuse does not implement FastKCNA's global layer refinement.
See [the supplied research examination](../search-performance/RESEARCH.md) for the
larger algorithms still awaiting implementation and qualification.

## TDD record

- `6996c59`: new both-norm primitive parity and mutable flat-batch coverage.
  The VPS compile failed because `cosineWithNorms` did not exist.
- `49cdee1`: concurrent read-work and cache-lifetime tests precede the runtime
  cache change. Main produced 9,010,221 reads, failing the <8,000,000 requirement.
- `8dee1bf`: warmed the existing SIMD scorer before the lifetime test's raw-bit
  comparisons. Main itself differed by one ULP between cold and warmed execution;
  after warmup, only the intended read-work test failed.
- `40f36c0`: first flat implementation passed correctness but failed performance.
- `37fec5c`: concurrent worker cache passed the focused tests and reduced the
  fixed probe to 6,811,334 reads. It subsequently failed the construction speed gate on both published corpora.
- `a9570da`: second flat implementation uses the direct cosine dot loop and
  dispatches outside the scan. Qualification results are recorded separately.

- `3fd67d1`: removed the rejected cache; extracted the original concurrent drain
  unchanged and added its allocation gate. Exact tie/order tests passed; the new
  allocation test failed at 65,624 bytes for 4,096 results.
- `25f47f8`: switched that drain to the existing exact primitive. Allocation,
  ties, append and executor-ownership tests passed before throughput measurements.

Persistent replay uses the unchanged benchmark and recall probe in
`vectors-bench/src/main/java/com/integrallis/vectors/bench`. It includes ordinary
commits, persistence and reopening; it does not invoke an embedding service.
The 100,000 × 512 DBpedia replay uses four commits, eight workers and 1,000
in-corpus recall probes. Level seeds in the ordinary collection API are generated
from time, so only the graph-only experiment provides fixed-seed construction.

```sh
python3 scripts/exact-reuse/run-replay.py BASELINE_JARS HNSW_JARS SOURCE_CHECKOUT \
  SAVED_COLLECTION OUTPUT --count 100000 --cadence 25000 --threads 8 --queries 1000 \
  --baseline-sha BASELINE_SHA --candidate-sha HNSW_SHA
```
