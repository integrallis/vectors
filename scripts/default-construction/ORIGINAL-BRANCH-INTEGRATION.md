# Original branch integration — 2026-10-03

## Scope and preservation

The original branch is `fix/commit-write-and-memory`. Its three commits through
`4bc505b` were already ancestors of `main` through PR #86. The remaining work was
46 modified or untracked source files, reviewed individually below.

The complete original source is preserved in local branch
`backup/original-work-before-integration-20261003`, commit `faab491e38422bd697a37f71678e12601c29f21d`.
A self-contained Git bundle, SHA-256 manifest, original patch and untracked-source
copies are retained under the local `vectors-bench/results/construction-20261003/original-branch-integration/`.
Existing measurement archives were retained. They are now ignored to prevent
accidentally committing generated collections, frozen jars or recursive source backups.

## Integrated changes

- Discard the writer graph and locks after a failed commit, on empty compaction,
  after refresh and on normal close. Successful append continues to reuse its graph,
  lock table and executor.
- Protect insertion queries and restored edge scores when a vector source reuses its
  return buffer. Allocate one scratch array per worker and one per rescore pass,
  instead of the original experiment's per-row clones. Stable-array and segment
  insertion paths keep their existing scoring behavior.
- Adapt the original persistence/recovery tests to default full-precision construction.
  Add explicit graph/lock/executor reuse and refresh coverage.
- Remove the temporary global phase counters and timing calls from runtime code.
  The manual cadence/recall benchmark remains, reporting total time and disk usage.

No SQ8 construction option or approximate scorer is added. Those experiments were
rejected in the earlier campaign and are recoverable from the backup. The retained
#86 default performance implementation stays in place.

## Reproduction

```sh
./gradlew :vectors-db:test --tests '*WriterGraphLifecycleTest' \
  :vectors-hnsw:test --tests '*SharedVectorBuildTest' --max-workers=4
./gradlew :vectors-hnsw:test :vectors-bench:recallGate --max-workers=4
```

Tests were run before the fixes: shared-buffer build/append/rescore checks failed,
as did graph/lock cleanup after failed publication, empty compaction, refresh and
close. The persistent append/reopen fixture also missed a stored vector. After
applying the fixes, all 21 focused cases passed, including ten failed-publication
retry schedules, unchanged exact graph comparisons and successful resource reuse.
The refresh fixture was corrected to publish the remote CURRENT pointer before
its failing cleanup assertion was recorded. Tests use a full-size beam for small
persistence fixtures; the existing recall gate and thresholds are unchanged.

The broader local run passed 149 HNSW cases, 45 database cases and the existing
recall gate (195 total). SpotBugs passed for both changed runtime modules. Its first
attempt found a pre-existing malformed directory tree with newline-containing names
in the original checkout; that tree was preserved outside the Java source root,
and analysis then passed. [Red/green and regression logs](evidence/original-branch/)
retain both the initial failure and successful checks.

These are correctness and resource-lifecycle fixes; they do not establish a new
throughput improvement. #86's benchmark results remain scoped to its measured source.

## File-by-file disposition

| Original path | Evaluation |
|---|---|
| `scripts/construction/.gitignore` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/README.md` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/article_report.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/build-reference.sh` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/compare.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/figures.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/hnswlib-reference.cpp` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/report.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/test_article_report.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/test_compare.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/test_reference.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/construction/test_report.py` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `scripts/run-construction-benchmark.sh` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-bench/build.gradle.kts` | Superseded by the default replay harness merged in #86; archive SQ8 benchmark variant. |
| `vectors-bench/src/jmh/java/com/integrallis/vectors/bench/SevenBitDistanceBenchmark.java` | Archive unused SQ8-specific kernel/API experiment; retain #86 default cosine optimization. |
| `vectors-bench/src/main/java/com/integrallis/vectors/bench/CollectionRecallProbe.java` | Already merged in #86. |
| `vectors-bench/src/main/java/com/integrallis/vectors/bench/CollectionReplayBenchmark.java` | Superseded by the default replay harness merged in #86; archive SQ8 benchmark variant. |
| `vectors-bench/src/main/java/com/integrallis/vectors/bench/ConstructionBenchmark.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-bench/src/main/java/com/integrallis/vectors/bench/dataset/BigAnnLoader.java` | Already merged in #86. |
| `vectors-bench/src/main/java/com/integrallis/vectors/bench/dataset/DatasetDownloader.java` | Already merged in #86. |
| `vectors-bench/src/main/java/com/integrallis/vectors/bench/dataset/DatasetRegistry.java` | Already merged; remaining difference is formatting. |
| `vectors-bench/src/main/java/com/integrallis/vectors/bench/dataset/Hdf5Loader.java` | Already merged; remaining difference is formatting. |
| `vectors-bench/src/test/java/com/integrallis/vectors/bench/BigAnnLoaderTest.java` | Already merged in #86. |
| `vectors-bench/src/test/java/com/integrallis/vectors/bench/CollectionReplayBenchmarkTest.java` | Superseded by the default replay harness merged in #86; archive SQ8 benchmark variant. |
| `vectors-bench/src/test/java/com/integrallis/vectors/bench/Hdf5LargeDatasetTest.java` | Already merged; remaining difference is formatting. |
| `vectors-core/src/main/java/com/integrallis/vectors/core/PanamaVectorUtilSupport.java` | Archive unused SQ8-specific kernel/API experiment; retain #86 default cosine optimization. |
| `vectors-core/src/main/java/com/integrallis/vectors/core/VectorUtil.java` | Archive unused SQ8-specific kernel/API experiment; retain #86 default cosine optimization. |
| `vectors-core/src/main/java/com/integrallis/vectors/core/VectorUtilSupport.java` | Archive unused SQ8-specific kernel/API experiment; retain #86 default cosine optimization. |
| `vectors-core/src/test/java/com/integrallis/vectors/core/SevenBitDistanceTest.java` | Archive unused SQ8-specific kernel/API experiment; retain #86 default cosine optimization. |
| `vectors-db/src/main/java/com/integrallis/vectors/db/VectorCollectionBuilder.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-db/src/main/java/com/integrallis/vectors/db/VectorCollectionConfig.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-db/src/main/java/com/integrallis/vectors/db/VectorCollectionImpl.java` | Integrate failed-commit, empty-compaction and close cleanup for the default exact graph; discard SQ8 calibration plumbing. |
| `vectors-db/src/main/java/com/integrallis/vectors/db/index/HnswIndexAdapter.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-db/src/test/java/com/integrallis/vectors/db/Sq8ConstructionCollectionTest.java` | Adapt persistence, retry, lifecycle, upsert/delete and compaction coverage to the default exact implementation. |
| `vectors-db/src/test/java/com/integrallis/vectors/db/index/HnswConstructionCacheTest.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/BuildScorer.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/ConcurrentHnswGraphBuilder.java` | Integrate shared-buffer query/edge-score protection, using reusable scratch; discard approximate scorer hooks. |
| `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/HnswGraphBuilder.java` | Keep #86 exact cache, ordering and serial shared-buffer fixes; archive approximate scorer hooks. |
| `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/HnswGraphDiagnostics.java` | Already merged in #86. |
| `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/HnswIndex.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/NeighborSelector.java` | Keep #86 exact cache, ordering and serial shared-buffer fixes; archive approximate scorer hooks. |
| `vectors-hnsw/src/main/java/com/integrallis/vectors/hnsw/Sq8BuildScorer.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-hnsw/src/test/java/com/integrallis/vectors/hnsw/BuildScorerTest.java` | Archive SQ8 experiment/support tooling; no new construction opt-in or unqualified default promotion. |
| `vectors-hnsw/src/test/java/com/integrallis/vectors/hnsw/HnswGraphDiagnosticsTest.java` | Already merged in #86. |
| `vectors-quantization/src/main/java/com/integrallis/vectors/quantization/ScalarQuantizer.java` | Already merged in #86. |
| `vectors-quantization/src/test/java/com/integrallis/vectors/quantization/ScalarQuantizerRepeatedValuesTest.java` | Already merged with the unit-test tag added. |
