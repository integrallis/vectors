# Exact reuse experiments

Baseline: merged main `212ad282d719a985706b8024ad310c73e5e85430`.

This round separates five hypotheses:

1. Heap COSINE batches can compute each stored row's squared norm once per scan,
   then reuse it across queries. Query norms are already reused on main. No norms
   persist across calls; mutations remain visible. Batches below four queries,
   short vectors and scalar execution retain the original scan. The two-query
   shape regressed in the first full comparison and is excluded automatically.
2. **Rejected:** concurrent HNSW diversity pruning can use the existing exact symmetric pair
   cache independently in each worker. Full ordinal keys distinguish collisions;
   each cache belongs to one vector source, metric and build/append operation.
   Cache storage is 192 KiB plus array headers per active worker. M=16,
   efConstruction=200 and query ef remain unchanged. It reduced a small fixed
   probe's reads but regressed construction time; it is absent from the final runtime.
3. Concurrent HNSW beam results can use the already-qualified exact heap drain,
   removing two temporary arrays and quadratic duplicate scans. Tie order remains
   identical. This is measured independently of both flat scoring and the rejected cache.
4. Uncached diversity pruning filters distinct candidate IDs into disjoint accepted
   and blocked sets. Its output insertion can omit another duplicate scan while
   retaining the same binary insertion position and tie order. Public insertion
   and serial cached pruning keep their duplicate protection. Both this change
   alone and its combination with the direct drain are measured.

5. The public collection API previously dispatched one independent search per
   query and never reached the optimized flat batch scan. Eligible in-memory
   COSINE batches now share scans across CPU workers, with at least four queries
   per worker. Eligibility is automatic; cache-enabled, normalized, filtered,
   mixed-k, persistent and smaller batches retain their existing execution.

None of these changes introduces a runtime option. Separate frozen jar directories
provide the experimental ablation; applications use the qualified defaults.

## Gates fixed before measurements

- Red tests precede implementation. Flat scores retain exact warmed float bits,
  IDs and mutation behavior; scalar/128-bit SIMD are checked independently.
- Flat batch median time must improve by at least 5% on both GLOVE COSINE and the
  retained 512-dimensional COSINE collection. Query counts 1, 2, 3, 4, 16 and 64,
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
[published input manifest](../search-performance/evidence/vps-20261003/published-inputs.json),
including original HDF5 URLs and SHA-256 hashes. Select the first 100,000 GLOVE
training rows, all 60,000 Fashion training rows, and the first 256 official test
rows from each, in source order. Convert to little-endian float32 without
normalizing or shuffling. An fbin begins with little-endian int32 row and dimension counts,
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

- `7426579`: distinct-neighbor insertion tests precede implementation; VPS test
  compilation failed because the primitive was absent. They cover order, ties,
  signed zero, raw score bits, full capacity and unchanged public duplicate checks.
- `9eda93d`: retain the exact original flat scan method for batches below four.
  The v2 three-pair result regressed at two queries despite improving large batches.
- `79d218a`: add an in-process double-reference numerical test alongside the strict
  subprocess bit test. This fixes CI coverage without changing its 80% threshold.
- `724db8d`: implement known-distinct insertion only in uncached pruning. Focused
  graph tests and the complete core suite/coverage gate pass on the VPS.

The unique-only ablation builds `724db8d` with just
`ConcurrentHnswGraphBuilder.java` restored from `212ad282`, then overlays its HNSW
jar on baseline jars. Binary inspection confirms only `NeighborArray` and
`NeighborSelector` differ. The combined HNSW arm adds only
`ConcurrentHnswGraphBuilder`; the flat arm changes only its adapter and three core
classes. `ablations.json` records every jar hash and changed class.

Construction runs now retain an encoded graph snapshot after measurements and
verify its SHA-256. The replay harness accepts `--recall-only` to resume quality
checks from completed timing records, while verifying that frozen inputs and
runtimes are unchanged. A relative-path harness error in the first replay was
fixed this way; its original timing records and failed recall log are retained.

## Public API integration

The adapter benchmark alone did not establish a gain for
`VectorCollection.searchBatch`. A JFR worker-count test was committed before
changing production routing. After correcting a typed-metadata fixture compile
error, the old path failed with 32 workers for 32 queries on the eight-processor
VPS; its three functional tests passed. Commit `7df8519` groups compatible
queries into eight scans, and all four tests pass. Each worker preserves request
order, projections, cutoffs and tombstones. One retained generation remains pinned
until all tasks finish, including exceptional paths. The original single-query
implementation remains unchanged.

```sh
python3 scripts/exact-reuse/run-flat.py BASELINE_JARS FINAL_JARS \
  PUBLISHED_INPUTS SAVED_COLLECTION OUTPUT --public-api \
  --query-counts 1,4,16,32,64 --rounds 3 \
  --baseline-sha 212ad282 --candidate-sha 7df8519

python3 scripts/exact-reuse/run-replay.py BASELINE_JARS FINAL_JARS SOURCE_TREE \
  SAVED_COLLECTION OUTPUT --rounds 3 --threads 8 \
  --baseline-sha 212ad282 --candidate-sha 7df8519
```

The public API gate, recorded before implementation, requires at least 5% less
median time at 64 queries on both COSINE inputs, identical warmed output digests,
three alternating fresh-JVM pairs, and disclosure of small-batch, individual-search
and other-metric controls. Published query inputs are held out; the retained
DBpedia flat probes are deterministic perturbed source rows, not held-out recall.
The persistent replay uses 1,000 in-corpus recall queries and reports that scope.

Measured results, controls and limitations: [RESULTS.md](RESULTS.md).

### Longer-warmup control investigation

The initial public run used the same eight-pass warmup as the adapter experiments.
It passed the Q64 speed gate on both COSINE corpora and every digest check, but
controls included a 16.43% slowdown and the unchanged GLOVE single-query path
varied from about 2.98 to 4.66 ms across baseline JVMs. That variability prompted
a separately recorded investigation, not deletion of the initial measurements.
`--public-api --warmup-seconds 3` adds a fixed minimum three seconds **per case**
to the existing eight-pass minimum, for both arms, with the same query counts,
three alternating pairs and seven measurement samples. `WARMUP` lines retain
actual counts and durations. The runtime source remains `7df8519` throughout.
Both complete result sets and the prior decision record are retained.

### Corpus placement before timing

The longer-warmup run exposed a 2.25× GLOVE single-search anomaly. Diagnostic
JFR/GC runs reproduced a similar slowdown on baseline as well, after young GC
moved corpus rows. The final public harness now calls `System.gc()` once after
building the collection and before timing, identically for both arms. This
settles corpus placement; the implementation itself never requests GC. The
initial and longer-warmup results, GC logs and decision record remain in
[evidence](evidence). Reproduce the final protocol with `--public-api
--warmup-seconds 3` and the same commands above. Use `audit-flat.py OUTPUT` to
independently recompute all flat medians, digest parity and the two Q64 speed gates.
