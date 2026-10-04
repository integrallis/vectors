# Faster collection queries

Baseline: merged main `c43c3160232dd20cf4b2ebd67f44b68c83edf56c`.

Two automatic query optimizations, without changing graphs, search budgets or precision:

- Large heap flat COSINE single-query scans score independent rows using at most eight
  common fork/join workers, then feed the original bounded heap in ordinal order.
  This uses spare CPU cores to reduce latency and allocates a temporary
  `4 * rowCount` byte score buffer. It does not establish a reduction in CPU work.
  Dispatch starts at four million vector components. A shared permit bounds the
  score buffer to one expanded query; active-query accounting curtails further
  task splitting when independent queries compete. Expansion resumes after
  50 milliseconds without observed overlap, preventing repeated expansion at the
  start of successive concurrent bursts. Euclidean and DOT scans keep serial scoring. The public collection batch
  API automatically coordinates its existing query parallelism so its scans do
  not expand into more workers. All scopes release on success or failure.
- Persisted full-precision HNSW COSINE queries score neighbor rows using byte
  offsets in the backing segment, avoiding one segment-view object per neighbor.
  The batch kernel's query norm is prepared once per query. The four-row SIMD
  accumulation, scalar tails and score transform match the original kernel.
  Single-node, heap, scalar, short-vector, noncontiguous segment, custom and quantized scorers retain
  their original algorithms. The storage adapter supplies offsets automatically.

No application options are required. Norms never persist between queries, and no
row norm cache is introduced. Caller-owned flat vectors remain mutable between
queries. The single-row and fused batch kernels use different reduction orders;
their prepared norms must not be interchanged.

## Qualification protocol

- TDD: `8280599` adds the prepared-batch contract before implementation. The VPS
  compiler fails with missing methods; `aedc00b` implements the candidate and
  passes the focused tests. The flat norm-reuse candidate was subsequently rejected:
  v2 took 6.96% more median time on DBpedia across three unrestricted JVM pairs.
  `8374f2d` adds parallel flat tests before implementation (missing-method red);
  `bed834c` implements parallel scoring and passes the focused tests.
  Offset contracts are introduced in `b4cc211` before `4733b61` implements them.
  The missing batch-coordination API in `34156c0` fails compilation before
  `aea59d4` implements the scoped scheduler. The original unqualified heap HNSW
  norm-only path is removed in `7e7cbf8`.
  `3edfd38` adds a deterministic injected-clock burst-budget contract and an
  unaccelerated HNSW routing assertion; both fail before `63954f8` implements them.
  Exact warmed offset score bits have a separate subprocess test; storage, filters,
  query mutation, ties and portable fallbacks are tested.
- Freeze baseline/candidate runtime jars and retain their SHA-256 checksums.
- Run one process at a time on a dedicated VPS. Three alternating fresh-JVM pairs;
  identical heap sizes; corpus placement settled before timing in both arms;
  at least 3 seconds and 12 passes of warmup per case; seven measured samples.
- Require identical result IDs and score-bit digests on every workload. Frozen
  graph parity implies no change in recall for those tested queries. Published
  GLOVE/Fashion queries are held out; DBpedia queries are deterministic perturbed
  source samples and are not held-out recall evidence.
- Target at least 5% median query-time reduction on each of two COSINE corpora.
  Report all ef levels, storage paths, flat batch and Euclidean controls. No
  reduction in ef or relaxation of the existing recall gate is permitted.
  Concurrent/control median time must not regress by more than 5%; every fork and
  raw sample is retained so variability is visible.
- Full unit/coverage and CI gates must pass before merging. Preserve unsuccessful
  experiments and controls alongside accepted results.

The first pilot uses one pair; it is diagnostic, not the final qualification.
`mapped` in the diagnostic harness means native segment scoring with contiguous
rows, not file/page-fault latency. The separate `public` case reopens a complete
persistent collection through `VectorCollection` and measures real mapped queries.
Public flat cases use the ordinary in-memory `VectorCollection.search` and
`searchBatch` methods, with default cache settings.

## Reproduce

Requires Java 25 with the Vector API and the retained corpus archive. Archive
`vectors-exact-reuse-archive.tar.gz` has SHA-256
`e3990f8474946dd1cbbb514da9be08f5b00609821087a3f5b7ce7b72e2ed5f59`.
The earlier [input provenance](../search-performance/evidence/vps-20261003/published-inputs.json)
records published dataset sources and hashes; the private retained DBpedia corpus
and graph snapshots are in that archive. The input archive is required to
reproduce that corpus; the committed logs alone do not reconstruct it.

```sh
JAVA_HOME=/path/to/jdk25 ./gradlew -I scripts/exact-reuse/freeze.gradle \
  :vectors-db:freezePerf -PfreezeDir=/absolute/path/to/jars \
  --no-daemon --max-workers=4

JAVA_HOME=/path/to/jdk25 python3 scripts/query-performance/run.py \
  BASELINE_JARS CANDIDATE_JARS EXTRACTED_ARCHIVE/vectors-search OUTPUT \
  --rounds 3 --modes flat,heap,mapped,public \
  --baseline-sha c43c316 --candidate-sha CANDIDATE_SHA
```

The runner hashes jars, inputs, source and Java executable, retains exact commands
and raw samples, checks digest parity, and rechecks hashes after completion.
Latency summaries are the median of seven samples per JVM and then the median
across JVMs, expressed per query. Timing includes result materialization and
consumption. Steady-state measurements do not establish cold-start or cold-page
performance, nor a ranking against competing libraries.

## Diagnostic revisions retained

The first query-norm-only pilot passed exact output parity but regressed DBpedia
flat single-query time by 8.88%. Separating the SIMD loop and scalar tail (v2)
also failed: three unrestricted pairs measured 6.96% more median time on DBpedia,
with large between-JVM variability in both baseline and candidate. Those flat
runtime changes were removed. The later parallel implementation retains the
original single-row similarity kernels.

A diagnostic pinned to one CPU also changed the JVM's detected CPU count and GC
ergonomics. It is retained, but must not be substituted for the ordinary
eight-processor environment. Subsequent HNSW affinity runs explicitly preserve
`ActiveProcessorCount` from the unrestricted host while pinning the measured
serial query process. Parallel flat measurements use all available VPS CPUs.

Concurrent public batch cases at Q=4,16,32 and independent `search()` calls at
Q=4,16 expose contention and allocation costs. Single-query latency and concurrent
throughput are reported separately.

The parallel v3 pilot regressed Fashion Q32 by 104%; limiting expansion to one
scan in v4 still regressed it by 39%. Both are rejected measurements. Automatic
collection batch coordination in v5 removed that regression in the pilot. Three
v6 pairs still showed 5.82% more time on independent Fashion Q16 searches, missing
the 5% control limit. v7 adds a deterministic burst budget; the final repeated
comparison determines acceptance.

For a second actual public HNSW corpus, compile `BuildPublicFixture.java` alongside
`QueryPerformance.java` with the baseline classpath, then invoke it with the GLOVE
training fbin and a new output directory. It creates a 20k-row persistent collection
once with baseline jars. Pass that directory as `--public-glove` to the runner;
both arms reopen separate copies of the same frozen generation and use the
published held-out test vectors. The native-segment diagnostic remains separate.

`ALLOCATION` lines report calling-thread bytes per query after warmup. They cover
HNSW's serial work but omit flat worker allocation; do not interpret them as total
allocation for concurrent flat workloads.


The v7 burst-budget pilot still regressed independent Fashion Q16 queries by
11.83%. Expansion for Euclidean/DOT is therefore excluded: only COSINE has
qualified positive evidence on both target corpora. The original serial heap
loop remains inside `search`, preserving its compilation context as closely as
possible. Earlier all-metric single-query gains are not shipped claims.
