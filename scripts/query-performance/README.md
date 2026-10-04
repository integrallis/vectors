# Query norm reuse

Baseline: merged main `c43c3160232dd20cf4b2ebd67f44b68c83edf56c`.

Two default query optimizations, without changing graphs, search budgets or precision:

- Large heap flat single-query scans score independent rows in the bounded common
  fork/join pool, then feed the original heap in ordinal order. Ties and exact
  scoring are preserved. This uses more CPU cores per query and allocates a
  temporary `4 * rowCount` byte score buffer. Dispatch is automatic above four
  million vector components when the common pool has multiple workers. Small
  scans and the optimized COSINE batch route retain their original execution.
- Full-precision HNSW COSINE scoring prepares the **batch kernel's** query norm
  once per query, instead of per neighbor list. Heap and segment kernels keep
  their original dot/row-norm accumulation, tail and score transform. Single-node
  scoring keeps its original kernel. Shared-return-buffer sources without segment
  support and custom/quantized scorers retain their original route.

The array single-row and fused batch kernels have different reduction orders.
Their prepared norms must not be interchanged. No norm persists between queries,
no row norm cache is introduced, and each searcher owns its existing scratch.

## Qualification protocol

- TDD: `8280599` adds the prepared-batch contract before implementation. The VPS
  compiler fails with missing methods; `aedc00b` implements the candidate and
  passes the focused tests. The flat norm-reuse candidate was subsequently rejected:
  v2 took 6.96% more median time on DBpedia across three unrestricted JVM pairs.
  `8374f2d` adds parallel flat tests before implementation (missing-method red);
  `bed834c` implements parallel scoring and passes the focused tests.
  Exact warmed score bits have a separate subprocess
  test; storage, filters, query mutation, ties and portable fallbacks are tested.
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

Concurrent flat cases at Q=4,16,32 expose contention and allocation costs, rather
than treating reduced single-query latency as proof of greater throughput.
