# Vectors 0.1.35 post-release audit — before/after against 0.1.34

**Question.** 0.1.35 was published (tag `d88bc6fe`, 2026-11-13) carrying 25 `perf(...)` commits, a new
default centroid trainer for every IVF index (`BuoyIndex` now calls `EMeans.train(dataset, k, 5.0f, seed)`
instead of `KMeans.train(dataset, k, 50, seed)`, `vectors-ivf/.../BuoyIndex.java:73`) and unconditional
per-matmul latency counters (`VectorUtil.java:3798`/`3802` → `GgufBatchedMatmulKernel.recordLatency`).
None of it was measured against the released 0.1.34 before publication. This directory holds the
measurement, the defects found, and the verdict on a 0.1.36.

Everything below is **measured here** unless marked otherwise. Every number is read from a file under
`raw/` by `scripts/derive-vectors.py`, `scripts/derive-stair.py` or `scripts/derive-confirm.py`; nothing
is typed in.

## Host and method

- host: AMD EPYC 7R13 Processor / 2 / 8; java: openjdk version "25.0.5" 2026-10-20 LTS. AWS m6a.4xlarge (`i-053a03178e8008bc5`, us-east-1) for the before/after; a second,
  identical instance (`i-00702a64cc9989197`) for the HNSW bisect so the two never shared a CPU.
- Both tags checked out as detached worktrees (`v0.1.34` = `b575aad0`-era release, `v0.1.35` = `d88bc6fe`),
  built once; JMH via the build's own `:vectors-bench:jmh` task with `-Pjmh.resultFormat=JSON`,
  `fork=2 warmup=3 iterations=5` (overriding the annotated `@Fork(1)`), **interleaved per benchmark**
  (0.1.34 then 0.1.35 for each selector, two full rounds). Medians over all forks of both rounds
  (n = 20 per arm); the unit is JMH's own.
- Recall: `:vectors-bench:recallGate` in `verify` mode on 0.1.35 against the committed baseline and in
  `record` mode on both tags; `slowTest --tests '*SiftSmall*RecallTest'` on both tags with SIFT-small at
  the path `DatasetRegistry` resolves (`research/repos/jvector/siftsmall/`).
- **Published null.** The first JMH attempt produced zero results: it copied the invocation from the
  repository's own `experiments/benchmark-comparison.sh`, which passes `-Pjmh.includes="^Class.method$"`
  (JMH matches includes with `find()` against the fully-qualified name, so a leading `^` on a bare class
  name never matches) and `-Pjmh.output=json` + `build/reports/jmh` (the build reads `-Pjmh.resultFormat`
  and writes `vectors-bench/build/results/jmh/results.<ext>`, `vectors-bench/build.gradle.kts:97-103`).
  That script therefore never produced a number for anyone. The failed logs are in `raw/jmh-failed-attempt/`;
  the corrected phase is `scripts/vecjmh.sh`.

## Quality

| recall@10 row | committed baseline (2026-06, k-means) | v0.1.34 record | v0.1.35 record | 0.1.35 − 0.1.34 |
| --- | --- | --- | --- | --- |
| HNSW efSearch=128 | 0.9145 | 0.913 | 0.9135 | +0.0005 |
| HNSW efSearch=200 | 0.9635 | 0.9635 | 0.963 | -0.0005 |
| IVF_FLAT nprobe=8,oq=1.0 | 0.3615 | 0.3615 | 0.365 | +0.0035 |
| IVF_PQ nprobe=8,m=16,rescore=4,oq=1.0 | 0.305 | 0.305 | 0.307 | +0.0020 |
| IVF_PQ nprobe=8,m=16,rescore=4,oq=4.0 | 0.716 | 0.716 | 0.7335 | +0.0175 |
| VAMANA L=128 | 0.9705 | 0.9705 | 0.9705 | +0.0000 |
| VAMANA L=200 | 0.983 | 0.983 | 0.983 | +0.0000 |

`recallGate` verify on 0.1.35 against the committed baseline: **PASSED** (the gate's tolerance is 0.005 per row;
HNSW efSearch=128 sits 0.0010 under its 2026-06 baseline on this host, inside tolerance; improvements never fail).
The e-means default moves IVF recall *up* at the gate's 10k×128 scale, most at IVF_PQ with over-query 4.0.
Note the gate was **restored from the Gradle build cache** in both release runs (43489444540, 43491441459)
and in the push CI of the release commit (43488583200); the executing run was CI 43480735198 on `d9fe7c34`
(same library bytes). A gate that can be satisfied from cache is addressed in the follow-up branch.

SIFT-small recall floors v0.1.34: tests=8 failures=0  ['fullPrecision_beamWidth128_recall10_above95()', 'fullPrecision_beamWidth200_recall10_above97()', 'fullPrecision_beamWidth64_recall10_above90()', 'fullPrecision_ef128_recall10_above95()', 'fullPrecision_ef200_recall10_above98()', 'fullPrecision_ef64_recall10_above90()', 'twoPassSq8_beamWidth128_recall10_above93()', 'twoPassSq8_ef128_recall10_above93()']
SIFT-small recall floors v0.1.35: tests=8 failures=0  ['fullPrecision_beamWidth128_recall10_above95()', 'fullPrecision_beamWidth200_recall10_above97()', 'fullPrecision_beamWidth64_recall10_above90()', 'fullPrecision_ef128_recall10_above95()', 'fullPrecision_ef200_recall10_above98()', 'fullPrecision_ef64_recall10_above90()', 'twoPassSq8_beamWidth128_recall10_above93()', 'twoPassSq8_ef128_recall10_above93()']

## Performance, phase 1 (same shapes as the repository's benchmark list, plus flat/IVF search and IVF build)

| benchmark | mode | v0.1.34 median [min..max] | v0.1.35 median [min..max] | 0.1.35 / 0.1.34 | reading |
| --- | --- | --- | --- | --- | --- |
| AnnSearchBenchmark.searchFlat{dim=128,k=10,n=2000} | thrpt (ops/ms) | 21.951 [17.370..22.393] (n=20) | 22.262 [22.156..22.506] (n=20) | 1.014 | faster by 1.4% |
| AnnSearchBenchmark.searchHnsw{dim=128,k=10,n=2000} | thrpt (ops/ms) | 24.250 [23.296..25.570] (n=20) | 23.559 [22.761..23.920] (n=20) | 0.971 | slower by 2.9% |
| AnnSearchBenchmark.searchIvfFlat{dim=128,k=10,n=2000} | thrpt (ops/ms) | 23.096 [22.444..24.505] (n=20) | 23.654 [22.588..25.114] (n=20) | 1.024 | faster by 2.4% |
| AnnSearchBenchmark.searchVamana{dim=128,k=10,n=2000} | thrpt (ops/ms) | 19.145 [18.401..19.410] (n=20) | 21.595 [20.468..22.136] (n=20) | 1.128 | faster by 12.8% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=100,liveSize=100000} | avgt (us/op) | 14.186 [13.917..14.353] (n=20) | 14.095 [13.574..14.692] (n=20) | 0.994 | faster by 0.6% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=100,liveSize=10000} | avgt (us/op) | 10.185 [9.929..10.719] (n=20) | 10.063 [9.766..10.286] (n=20) | 0.988 | faster by 1.2% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=100,liveSize=1000} | avgt (us/op) | 8.263 [8.213..8.446] (n=20) | 8.354 [8.129..8.942] (n=20) | 1.011 | slower by 1.1% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=1000,liveSize=100000} | avgt (us/op) | 142.442 [138.326..148.435] (n=20) | 138.086 [132.857..147.649] (n=20) | 0.969 | faster by 3.1% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=1000,liveSize=10000} | avgt (us/op) | 97.049 [92.056..103.232] (n=20) | 96.114 [94.360..104.767] (n=20) | 0.990 | faster by 1.0% |
| CommitPipelineBenchmark.copyOfAndStage{batchSize=1000,liveSize=1000} | avgt (us/op) | 80.208 [77.707..84.011] (n=20) | 80.369 [77.932..83.524] (n=20) | 1.002 | slower by 0.2% |
| GgufBatchedMatmulBenchmark.batched{batchSize=1,cols=2048,rows=1024} | avgt (ms/op) | 0.089 [0.089..0.092] (n=20) | 0.089 [0.089..0.092] (n=20) | 1.001 | slower by 0.1% |
| GgufBatchedMatmulBenchmark.batched{batchSize=2,cols=2048,rows=1024} | avgt (ms/op) | 0.271 [0.216..0.274] (n=20) | 0.247 [0.216..0.275] (n=20) | 0.911 | faster by 8.9% |
| GgufBatchedMatmulBenchmark.batched{batchSize=24,cols=2048,rows=1024} | avgt (ms/op) | 1.525 [1.523..1.632] (n=20) | 1.526 [1.522..1.585] (n=20) | 1.001 | slower by 0.1% |
| GgufBatchedMatmulBenchmark.batched{batchSize=32,cols=2048,rows=1024} | avgt (ms/op) | 2.027 [2.021..2.239] (n=20) | 2.009 [2.002..2.085] (n=20) | 0.991 | faster by 0.9% |
| GgufBatchedMatmulBenchmark.batched{batchSize=4,cols=2048,rows=1024} | avgt (ms/op) | 0.300 [0.299..0.310] (n=20) | 0.300 [0.299..0.321] (n=20) | 1.000 | slower by 0.0% |
| GgufBatchedMatmulBenchmark.batched{batchSize=8,cols=2048,rows=1024} | avgt (ms/op) | 0.546 [0.543..0.584] (n=20) | 0.545 [0.543..0.547] (n=20) | 0.999 | faster by 0.1% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=1,cols=2048,rows=1024} | avgt (ms/op) | 0.107 [0.103..0.122] (n=20) | 0.105 [0.104..0.110] (n=20) | 0.982 | faster by 1.8% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=2,cols=2048,rows=1024} | avgt (ms/op) | 0.240 [0.239..0.251] (n=20) | 0.242 [0.239..0.279] (n=20) | 1.005 | slower by 0.5% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=32,cols=2048,rows=1024} | avgt (ms/op) | 2.417 [2.408..2.576] (n=20) | 2.432 [2.412..2.581] (n=20) | 1.006 | slower by 0.6% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=4,cols=2048,rows=1024} | avgt (ms/op) | 0.307 [0.305..0.317] (n=20) | 0.320 [0.307..0.334] (n=20) | 1.045 | slower by 4.5% |
| GgufQ4KBatchedMatmulBenchmark.batched{batchSize=8,cols=2048,rows=1024} | avgt (ms/op) | 0.614 [0.605..0.626] (n=20) | 0.611 [0.604..0.643] (n=20) | 0.995 | faster by 0.5% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=1,cols=2048,rows=1024} | avgt (ms/op) | 0.104 [0.103..0.114] (n=20) | 0.105 [0.104..0.121] (n=20) | 1.004 | slower by 0.4% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=2,cols=2048,rows=1024} | avgt (ms/op) | 0.208 [0.207..0.219] (n=20) | 0.209 [0.208..0.215] (n=20) | 1.001 | slower by 0.1% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=32,cols=2048,rows=1024} | avgt (ms/op) | 3.425 [3.329..3.868] (n=20) | 3.419 [3.322..3.448] (n=20) | 0.998 | faster by 0.2% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=4,cols=2048,rows=1024} | avgt (ms/op) | 0.418 [0.416..0.436] (n=20) | 0.427 [0.418..0.497] (n=20) | 1.021 | slower by 2.1% |
| GgufQ4KBatchedMatmulBenchmark.independent{batchSize=8,cols=2048,rows=1024} | avgt (ms/op) | 0.846 [0.838..1.303] (n=20) | 0.838 [0.828..0.858] (n=20) | 0.991 | faster by 0.9% |
| IndexBuildBenchmark.buildHnsw{dim=128,n=1000} | ss (ms/op) | 294.501 [287.027..302.203] (n=20) | 275.484 [269.298..294.647] (n=20) | 0.935 | faster by 6.5% |
| IndexBuildBenchmark.buildHnsw{dim=128,n=5000} | ss (ms/op) | 2699.921 [2638.292..2814.628] (n=20) | 2587.821 [2532.574..2627.190] (n=20) | 0.958 | faster by 4.2% |
| IndexBuildBenchmark.buildIvfFlat{dim=128,n=1000} | ss (ms/op) | 5.788 [5.167..6.407] (n=20) | 5.856 [5.261..8.465] (n=20) | 1.012 | slower by 1.2% |
| IndexBuildBenchmark.buildIvfFlat{dim=128,n=5000} | ss (ms/op) | 76.485 [75.210..79.696] (n=20) | 34.721 [33.854..37.400] (n=20) | 0.454 | faster by 54.6% |
| IndexBuildBenchmark.buildVamana{dim=128,n=1000} | ss (ms/op) | 1882.260 [1845.248..1932.869] (n=20) | 1865.335 [1826.777..1910.343] (n=20) | 0.991 | faster by 0.9% |
| IndexBuildBenchmark.buildVamana{dim=128,n=5000} | ss (ms/op) | 14323.668 [14151.466..14779.376] (n=20) | 14250.858 [14081.584..14561.041] (n=20) | 0.995 | faster by 0.5% |

Reading: Vamana search +12.8%, HNSW build 4–7% faster, IVF_FLAT build at n=5000 2.2× faster (e-means),
commit pipeline flat to 3% faster, GGUF batched matmul (the path `models` exercises) flat within noise
including with the unconditional `nanoTime` pair per call. **The one regression is `searchHnsw` at −2.9%**
(0.971; both rounds below 1, every 0.1.35 fork at or under the 0.1.34 median) — investigated below.

## Code read of the risky-looking changes (v0.1.34..v0.1.35, 68 files)

- `a5fbaef4` removed the defensive `clone()` of `nodeIds()/scores()` in the four HNSW/Vamana adapters.
  Safe: `HnswSearcher` (lines 534–541, 638–646, 983–989) and `VamanaSearcher` (375–387, 474–482) allocate
  fresh result arrays per search, and the only consumers of `SearchOutcome` arrays read them
  (`VectorCollectionImpl.java:2349-2350`, `IndexShards.java:181-182`).
- `29a28e0f` moved the single ingest clone to `maybeNormalizeForCosine` (now unconditional) and removed the
  second clone from `StagingBuffer`; every ingest path passes through the choke point. Safe.
- `25a38733` returns the stored `Document` when the projection is identical. The projected copy already
  shared the same vector array and metadata map, so nothing new is exposed. Safe.
- `dc0cff0e` (`SearchResult` without `List.copyOf`): the hits list is a fresh `ArrayList` per search
  wrapped in `Collections.unmodifiableList`. Safe.

## Defects in the published 0.1.35

1. **Missing upstream notice (compliance).** `EMeans.java` declares itself "a faithful port of the FAISS
   NumPy CPU path (contrib/e_means/e_means.py)". FAISS is MIT-licensed (upstream `LICENSE`, verified
   2026-11-14; `contrib/e_means/e_means.py` carries "Copyright (c) Meta Platforms, Inc. and affiliates").
   The published class carries only the Integrallis Apache header; `META-INF/NOTICE` and `META-INF/licenses/`
   do not mention FAISS. MIT requires the notice in copies and substantial portions; the repository's own
   rule (java-ai `CLAUDE.md` §6) requires retaining it. Only a new release can carry the notice.
2. **No `## [0.1.35]` CHANGELOG section.** At the tag, `CHANGELOG.md` goes `[Unreleased]` → `[0.1.34]`;
   the 0.1.35 entries sit under `[Unreleased]`, and neither the e-means default (a behavioural change for
   every IVF index) nor any of the 25 `perf(...)` commits is mentioned.
3. **The comparison tooling never worked** (see *Published null* above), and 16 of the 23
   `experiments/perf-*/README.md` files contain no measurement — hypothesis and decision rule only.
   `experiments/perf-hnsw-reusable-scorer/README.md` states its own decision rule as "JMH shows no
   regression" and then records that no usable JMH run was made.
4. **A cacheable release gate.** `recallGate` (`build.gradle.kts:423`) has no `cacheIf`/`upToDateWhen`;
   see *Quality*.

## HNSW search "regression" — attribution, and the null it became

Phase 1 read `searchHnsw` (2000×128) at 0.971× throughput with every 0.1.35 fork at or under the 0.1.34
median, and phase 2 read `searchRawHnsw` 100k×128 at 1.080× latency with non-overlapping ranges. Three
runs on a second, identical instance (`i-00702a64cc9989197`) tried to attribute it to a commit.

### Staircase — 25 first-parent points, `searchHnsw` 2000×128, fork=2 × 2 rounds (`scripts/stair.sh`)

Round 1 walked the chain oldest→newest, round 2 newest→oldest.

| # | point | r1 forks | r2 forks | median | vs v0.1.34 | step | subject |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | v0.1.34 | 23.35, 24.60 | 24.59, 22.96 | 23.97 | 1.000 | +0.0% | Merge pull request #123 from integrallis/release/0.1.34 |
| 1 | 871dafcc | 23.58, 23.70 | 24.15, 24.45 | 23.93 | 0.998 | -0.2% | perf(hnsw): eliminate per-search allocation in pickSeeds |
| 2 | ef814631 | 23.38, 22.90 | 24.48, 24.12 | 23.75 | 0.991 | -0.7% | perf(hnsw): reuse NeighborSelector scratch buffers on pruning path |
| 3 | 686864b4 | 23.47, 24.43 | 24.37, 22.92 | 23.92 | 0.998 | +0.7% | perf(builder): generation-stamped int[] visited set in concurrent buil |
| 4 | 73bcc1b8 | 23.92, 24.11 | 23.12, 23.88 | 23.90 | 0.997 | -0.1% | perf(searcher): size scratch heaps by graph degree instead of fixed 25 |
| 5 | 97579da4 | 24.32, 24.28 | 23.72, 24.14 | 24.21 | 1.010 | +1.3% | perf(neighbor): linear-scan insertion for small NeighborArray (capacit |
| 6 | a5fbaef4 | 23.32, 25.32 | 22.68, 22.99 | 23.16 | 0.966 | -4.4% | perf(adapter): remove unnecessary defensive array clones on search pat |
| 7 | d43220ee | 23.15, 24.68 | 23.94, 24.55 | 24.25 | 1.011 | +4.7% | perf(builder): reuse search-result arrays in concurrent graph builders |
| 8 | e128b7b5 | 22.97, 23.28 | 24.87, 23.40 | 23.34 | 0.974 | -3.7% | perf(search): cache ACORN pre-filter flag in Generation |
| 9 | d79e41be | 24.67, 23.37 | 23.01, 22.82 | 23.19 | 0.968 | -0.6% | perf(searcher): size rescore heap to 256 to avoid mid-rescore growth |
| 10 | 1b82abfd | 24.12, 24.03 | 24.19, 24.59 | 24.16 | 1.008 | +4.1% | perf(hnsw): generation-stamped int[] visited set + reusable result dra |
| 11 | d54794d2 | 24.03, 24.62 | 24.77, 23.65 | 24.32 | 1.015 | +0.7% | perf(hnsw): generation-stamped dedup for searchMultiStart merge |
| 12 | 07d02399 | 23.03, 24.62 | 24.63, 24.90 | 24.62 | 1.027 | +1.2% | perf(cache): eliminate varargs Object[] allocations in FilterHasher |
| 13 | e7fe344e | 23.13, 23.78 | 24.94, 22.26 | 23.46 | 0.978 | -4.7% | perf(hnsw): cap multi-start worker creation at availableProcessors |
| 14 | 0ba0417c | 24.12, 24.55 | 24.69, 24.42 | 24.49 | 1.021 | +4.4% | perf(cache): zero-allocation QvCache lookup on cache hit |
| 15 | dda9381a | 24.67, 23.47 | 24.25, 23.41 | 23.86 | 0.995 | -2.6% | perf(db): reuse thread-local scratch for cosine query normalization |
| 16 | 8b2d0b51 | 24.23, 24.86 | 23.74, 23.38 | 23.99 | 1.001 | +0.5% | perf(db): skip BitSet.clone() on append-only commits (#167) |
| 17 | 25a38733 | 23.73, 23.54 | 22.80, 23.47 | 23.50 | 0.980 | -2.0% | perf(db): reuse stored Document on search when projection is identical |
| 18 | ca2de0d9 | 24.57, 23.67 | 25.15, 24.02 | 24.30 | 1.014 | +3.4% | perf(db): normalize vectors outside the write lock on ingest |
| 19 | dc0cff0e | 22.84, 23.06 | 24.71, 23.75 | 23.41 | 0.976 | -3.7% | perf(db): eliminate List.copyOf allocation in SearchResult constructor |
| 20 | 29a28e0f | 23.33, 23.01 | 24.92, 23.53 | 23.43 | 0.978 | +0.1% | perf(db): eliminate double vector clone on ingest |
| 21 | b500e86e | 22.32, 24.40 | 24.56, 23.09 | 23.75 | 0.991 | +1.3% | perf(hnsw): eliminate per-search default-scorer allocation (#177) |
| 22 | b575aad0 | 24.35, 24.54 | 23.34, 22.64 | 23.85 | 0.995 | +0.4% | Merge pull request #181 from integrallis/docs/score-scale-explicit |
| 23 | 4b22ff0c | 24.84, 22.86 | 23.38, 24.36 | 23.87 | 0.996 | +0.1% | feat: expose raw scores via SearchResult.rawScoreOf() and SimilarityFu |
| 24 | v0.1.35 | 22.92, 24.54 | 23.75, 23.30 | 23.52 | 0.981 | -1.5% | docs: bump version pills and antora config to 0.1.35 |

No step stands out of the point-to-point swings (±4% with four forks per point).

### Confirm — four arms, `searchHnsw` + `buildHnsw`, fork=2 × 2 rounds (`scripts/confirm.sh`)

`fixA` = static nested scorer classes with final `sim`/`vectors` references (the b500e86e inner classes
read them through `HnswSearcher.this`); `fixB` = fixA plus the query copied into a final scratch array.
Patches in `raw/confirm/`.

AnnSearchBenchmark_searchHnsw{dim=128,k=10,n=2000}  (thrpt, ops/ms; per-fork medians, n per arm)
| arm | median | forks | vs v0.1.34 | vs v0.1.35 |
| --- | --- | --- | --- | --- |
| v0.1.34 | 23.951 | 24.34, 23.57, 23.49, 24.38 | 1.000 (+0.0%) | 1.012 (+1.2%) |
| v0.1.35 | 23.662 | 22.86, 24.16, 24.43, 23.17 | 0.988 (-1.2%) | 1.000 (+0.0%) |
| fixA | 23.868 | 24.02, 23.72, 24.65, 23.44 | 0.997 (-0.3%) | 1.009 (+0.9%) |
| fixB | 24.176 | 23.13, 24.77, 24.32, 24.03 | 1.009 (+0.9%) | 1.022 (+2.2%) |

IndexBuildBenchmark_buildHnsw{dim=128,n=1000}  (ss, ms/op; per-fork medians, n per arm)
| arm | median | forks | vs v0.1.34 | vs v0.1.35 |
| --- | --- | --- | --- | --- |
| v0.1.34 | 294.887 | 298.52, 293.44, 294.46, 295.31 | 1.000 (+0.0%) | 1.077 (-7.7%) |
| v0.1.35 | 273.788 | 268.51, 277.96, 272.29, 275.29 | 0.928 (+7.2%) | 1.000 (+0.0%) |
| fixA | 272.558 | 279.19, 271.21, 270.61, 273.91 | 0.924 (+7.6%) | 0.996 (+0.4%) |
| fixB | 276.144 | 269.19, 281.46, 272.37, 279.92 | 0.936 (+6.4%) | 1.009 (-0.9%) |

IndexBuildBenchmark_buildHnsw{dim=128,n=5000}  (ss, ms/op; per-fork medians, n per arm)
| arm | median | forks | vs v0.1.34 | vs v0.1.35 |
| --- | --- | --- | --- | --- |
| v0.1.34 | 2658.238 | 2693.66, 2634.12, 2643.14, 2673.33 | 1.000 (+0.0%) | 1.029 (-2.9%) |
| v0.1.35 | 2584.002 | 2566.21, 2687.99, 2581.11, 2586.90 | 0.972 (+2.8%) | 1.000 (+0.0%) |
| fixA | 2630.666 | 2573.27, 2668.39, 2663.01, 2598.32 | 0.990 (+1.0%) | 1.018 (-1.8%) |
| fixB | 2570.709 | 2609.49, 2554.19, 2562.11, 2579.31 | 0.967 (+3.3%) | 0.995 (+0.5%) |

Everything within ±2% on four forks; `buildHnsw` is 3–8% faster on every 0.1.35-based arm regardless.

### Attribution — `searchRawHnsw` 10k×128, 15 forks per arm (`scripts/revert.sh`)

v0.1.35 minus one commit each (`git revert --no-commit` on a v0.1.35 worktree; `73bcc1b8` conflicted and
was skipped), plus both tags and fixA. `fork=5 × 3 interleaved rounds`.

searchRawHnsw {dim=128,n=10000} (avgt us/op, lower is better; per-fork medians)
| arm | forks | median | min | max | vs v0.1.34 | vs v0.1.35 | P(fork faster than a v0.1.35 fork) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0.1.34 | 15 | 73.89 | 67.37 | 76.09 | 1.000 | 1.091 | 0.07 |
| v0.1.35 | 15 | 67.72 | 65.03 | 72.15 | 0.917 | 1.000 | 0.47 |
| fixA | 15 | 70.51 | 64.60 | 72.54 | 0.954 | 1.041 | 0.36 |
| rev-871dafcc | 15 | 67.28 | 65.52 | 72.16 | 0.911 | 0.993 | 0.51 |
| rev-97579da4 | 15 | 70.82 | 65.65 | 73.21 | 0.958 | 1.046 | 0.33 |
| rev-b500e86e | 15 | 67.34 | 64.48 | 72.74 | 0.911 | 0.994 | 0.53 |

**Reading.** With fifteen forks per arm, **0.1.35 is 0.917× the latency of 0.1.34 — 8% faster** — and
the probability that a 0.1.34 fork beats a 0.1.35 fork is 0.07. No revert moves the result:
`rev-871dafcc` and `rev-b500e86e` sit on 0.1.35 (0.993, 0.994); `rev-97579da4` and `fixA` are 4–5%
*slower* than 0.1.35, i.e. removing the linear neighbour insertion or changing the scorer costs. The
same build's fork medians span 65–76 µs (±6%), which is exactly the band the phase-1 (−2.9%, four
forks per arm) and phase-2 (+8%, one fork per arm per round) readings fell in. **The regression did not
survive statistical power; the fix branch is not merged.** The 100k×128 shape is being re-measured at
fifteen forks per arm (`scripts/power100k.sh`) so no claim about it rests on two forks.

A fork=5 × 3-round run at n=2000 and n=50000 (`scripts/power.sh`) was started and aborted after one of
nine runs to give the attribution run the box; its single artifact is in `raw/power-aborted/` and is not
used anywhere.

**Method lesson, recorded for the comparison script:** on this host a JMH fork-to-fork spread of ±6%
on a 70 µs search is normal; `fork=2 × 2 rounds` cannot resolve a 2–5% question, and single-fork cells
cannot resolve 8%. `experiments/benchmark-comparison.sh` now defaults to five forks.

## Phase 2 — facade and raw HNSW search, IVF search at 10k/100k × 128/768

`fork=1` (each parameter combination rebuilds its index in `@Setup`), 2 interleaved rounds, so n = 10 iterations
per cell and single-fork noise applies: read the ranges, not just the medians. Raw = `HnswIndex` directly;
facade = `VectorCollection.search` through `vectors-db`.

| benchmark | mode | v0.1.34 median [min..max] | v0.1.35 median [min..max] | 0.1.35 / 0.1.34 | reading |
| --- | --- | --- | --- | --- | --- |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=16} | avgt (us/op) | 38.694 [37.564..39.797] (n=10) | 42.878 [41.299..44.208] (n=10) | 1.108 | slower by 10.8% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=1} | avgt (us/op) | 2.791 [2.753..2.828] (n=10) | 3.141 [3.073..3.202] (n=10) | 1.125 | slower by 12.5% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=2} | avgt (us/op) | 4.375 [4.230..4.841] (n=10) | 5.708 [5.581..5.875] (n=10) | 1.305 | slower by 30.5% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=32} | avgt (us/op) | 77.270 [72.000..82.600] (n=10) | 79.073 [75.067..82.848] (n=10) | 1.023 | slower by 2.3% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=4} | avgt (us/op) | 9.944 [9.905..10.631] (n=10) | 12.192 [11.606..12.381] (n=10) | 1.226 | slower by 22.6% |
| IvfSearchBenchmark.searchIvf{dim=128,n=10000,nprobe=8} | avgt (us/op) | 19.256 [18.548..20.646] (n=10) | 25.546 [24.008..26.724] (n=10) | 1.327 | slower by 32.7% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=16} | avgt (us/op) | 152.927 [149.611..155.645] (n=10) | 146.979 [138.520..155.572] (n=10) | 0.961 | faster by 3.9% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=1} | avgt (us/op) | 10.788 [10.424..11.178] (n=10) | 10.162 [10.149..10.416] (n=10) | 0.942 | faster by 5.8% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=2} | avgt (us/op) | 18.944 [18.358..19.505] (n=10) | 20.596 [19.356..21.882] (n=10) | 1.087 | slower by 8.7% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=32} | avgt (us/op) | 280.281 [264.509..294.911] (n=10) | 287.802 [270.395..307.002] (n=10) | 1.027 | slower by 2.7% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=4} | avgt (us/op) | 35.562 [33.446..37.691] (n=10) | 37.177 [35.415..39.369] (n=10) | 1.045 | slower by 4.5% |
| IvfSearchBenchmark.searchIvf{dim=128,n=100000,nprobe=8} | avgt (us/op) | 73.533 [68.109..77.801] (n=10) | 74.422 [70.631..78.275] (n=10) | 1.012 | slower by 1.2% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=16} | avgt (us/op) | 219.413 [218.466..220.409] (n=10) | 167.619 [167.279..168.158] (n=10) | 0.764 | faster by 23.6% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=1} | avgt (us/op) | 16.816 [16.281..17.046] (n=10) | 11.083 [10.081..12.190] (n=10) | 0.659 | faster by 34.1% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=2} | avgt (us/op) | 32.599 [32.495..32.773] (n=10) | 17.962 [16.346..19.498] (n=10) | 0.551 | faster by 44.9% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=32} | avgt (us/op) | 470.357 [466.187..474.256] (n=10) | 331.358 [329.832..335.721] (n=10) | 0.704 | faster by 29.6% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=4} | avgt (us/op) | 70.855 [68.973..72.838] (n=10) | 42.276 [42.039..42.493] (n=10) | 0.597 | faster by 40.3% |
| IvfSearchBenchmark.searchIvf{dim=768,n=10000,nprobe=8} | avgt (us/op) | 155.658 [142.534..169.203] (n=10) | 89.601 [80.912..97.358] (n=10) | 0.576 | faster by 42.4% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=16} | avgt (us/op) | 715.988 [711.258..721.609] (n=10) | 769.478 [759.250..780.096] (n=10) | 1.075 | slower by 7.5% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=1} | avgt (us/op) | 42.907 [37.904..45.944] (n=10) | 52.290 [48.020..56.714] (n=10) | 1.219 | slower by 21.9% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=2} | avgt (us/op) | 92.055 [91.697..92.340] (n=10) | 93.471 [91.698..96.422] (n=10) | 1.015 | slower by 1.5% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=32} | avgt (us/op) | 2187.297 [2163.554..2208.196] (n=10) | 2118.924 [2088.733..2152.516] (n=10) | 0.969 | faster by 3.1% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=4} | avgt (us/op) | 155.073 [153.520..156.480] (n=10) | 172.825 [161.399..184.629] (n=10) | 1.114 | slower by 11.4% |
| IvfSearchBenchmark.searchIvf{dim=768,n=100000,nprobe=8} | avgt (us/op) | 323.571 [322.751..324.790] (n=10) | 308.268 [306.088..313.107] (n=10) | 0.953 | faster by 4.7% |
| VectorDbBenchmark.searchFacadeHnsw{dim=128,n=100000} | avgt (us/op) | 118.398 [117.326..118.654] (n=10) | 117.692 [116.581..118.725] (n=10) | 0.994 | faster by 0.6% |
| VectorDbBenchmark.searchFacadeHnsw{dim=128,n=10000} | avgt (us/op) | 73.691 [72.739..74.175] (n=10) | 73.863 [70.729..77.756] (n=10) | 1.002 | slower by 0.2% |
| VectorDbBenchmark.searchFacadeHnsw{dim=768,n=100000} | avgt (us/op) | 323.950 [320.934..328.418] (n=10) | 266.556 [263.806..269.307] (n=10) | 0.823 | faster by 17.7% |
| VectorDbBenchmark.searchFacadeHnsw{dim=768,n=10000} | avgt (us/op) | 221.610 [207.237..236.024] (n=10) | 235.341 [234.083..237.335] (n=10) | 1.062 | slower by 6.2% |
| VectorDbBenchmark.searchRawHnsw{dim=128,n=100000} | avgt (us/op) | 114.136 [111.868..116.429] (n=10) | 123.321 [122.217..125.075] (n=10) | 1.080 | slower by 8.0% |
| VectorDbBenchmark.searchRawHnsw{dim=128,n=10000} | avgt (us/op) | 70.852 [67.230..74.405] (n=10) | 69.006 [67.026..71.295] (n=10) | 0.974 | faster by 2.6% |
| VectorDbBenchmark.searchRawHnsw{dim=768,n=100000} | avgt (us/op) | 302.278 [278.349..326.028] (n=10) | 301.066 [277.933..318.757] (n=10) | 0.996 | faster by 0.4% |
| VectorDbBenchmark.searchRawHnsw{dim=768,n=10000} | avgt (us/op) | 239.692 [234.984..243.879] (n=10) | 204.358 [202.857..206.232] (n=10) | 0.853 | faster by 14.7% |

Reading. **`searchRawHnsw` at dim = 128, n = 100k is 1.080× latency with non-overlapping ranges** —
the regression phase 1 saw on a 2000-node graph, at scale; at 10k the two rounds disagree (+5.5% then
−2.6%), which is what one fork per cell looks like. dim = 768 is faster raw at 10k (0.853×) and flat at
100k; the facade is flat at dim = 128 and 0.823× at 768×100k. IVF search latency at a *fixed* nprobe
swings −9% to +24% (128) and −50% to +20% (768) in both directions: the e-means partitions differ from
the k-means ones, so the same nprobe scans different cluster sizes. That is not a speed verdict either
way; the recall-versus-latency curves below are the comparison that holds.

## Phase 3 — the e-means default's training cost against the k-means it replaced

`KMeansEMeansBenchmark` exists only from 0.1.35; it trains both algorithms on the same data on the same
host, with the iteration counts `BuoyIndex` used before and after (`KMeans.train(…, 50, …)` vs
`EMeans.train(…, 5.0f, …)`). Annotated `fork=1 warmup=2 iterations=3`.

| n | dim | k | trainKMeans ms | trainEMeans ms | e-means / k-means |
| --- | --- | --- | --- | --- | --- |
| 10,000 | 64 | 64 | 171.5 | 32.4 | 0.189 |
| 10,000 | 64 | 256 | 285.6 | 120.9 | 0.423 |
| 10,000 | 64 | 1024 | 1113.2 | 433.6 | 0.390 |
| 10,000 | 128 | 64 | 186.3 | 58.0 | 0.311 |
| 10,000 | 128 | 256 | 390.0 | 214.1 | 0.549 |
| 10,000 | 128 | 1024 | 1170.4 | 835.5 | 0.714 |
| 50,000 | 64 | 64 | 864.9 | 231.3 | 0.267 |
| 50,000 | 64 | 256 | 3282.5 | 820.6 | 0.250 |
| 50,000 | 64 | 1024 | 9284.5 | 3384.0 | 0.364 |
| 50,000 | 128 | 64 | 1400.3 | 456.3 | 0.326 |
| 50,000 | 128 | 256 | 5260.5 | 1697.2 | 0.323 |
| 50,000 | 128 | 1024 | 14390.3 | 6795.8 | 0.472 |
| 100,000 | 64 | 64 | 1773.3 | 561.0 | 0.316 |
| 100,000 | 64 | 256 | 6664.7 | 2101.4 | 0.315 |
| 100,000 | 64 | 1024 | 26251.8 | 8183.4 | 0.312 |
| 100,000 | 128 | 64 | 3682.3 | 1021.2 | 0.277 |
| 100,000 | 128 | 256 | 11955.5 | 3856.2 | 0.323 |
| 100,000 | 128 | 1024 | 53632.8 | 15246.8 | 0.284 |

ratio range 0.189..0.714, median 0.319 over 18 shapes

## IVF recall versus latency — both tags, identical corpus

`TieredIvfRecallQpsBenchmark` (`BuoyIndex` routing, the path whose trainer changed) on its own
mixture-of-Gaussians corpus (16 centres, seed `0xC0FFEE`), nprobe 1–64, two rounds per tag, medians.
Both tags see the same vectors and queries; only the trainer differs (`KMeans.train(…, 50, …)` vs
`EMeans.train(…, 5.0f, …)`). Two partitions are only comparable at equal recall, so each table is
followed by 0.1.35's latency **interpolated at 0.1.34's recall** along 0.1.35's own curve.

### Tiered IVF (BuoyIndex) synthetic mixture-of-Gaussians, dim=128 n=200,000, recall@10 vs median latency (us), both rounds
| nprobe | recall 0.1.34 | recall 0.1.35 | Δrecall | latency 0.1.34 | latency 0.1.35 | 0.1.35/0.1.34 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1087 | 0.0966 | -0.0121 | 40.9 | 37.5 | 0.916 |
| 2 | 0.1898 | 0.1712 | -0.0186 | 75.8 | 67.8 | 0.894 |
| 4 | 0.3134 | 0.2873 | -0.0261 | 141.9 | 129.2 | 0.911 |
| 8 | 0.4885 | 0.4552 | -0.0333 | 275.1 | 250.1 | 0.909 |
| 16 | 0.6921 | 0.6682 | -0.0239 | 521.5 | 485.8 | 0.931 |
| 32 | 0.8309 | 0.8271 | -0.0038 | 949.5 | 935.0 | 0.985 |
| 64 | 0.9038 | 0.9085 | +0.0047 | 1852.8 | 1849.0 | 0.998 |
    v0.1.34: built tiered IVF (k=448) in 583.45s
    v0.1.34: built tiered IVF (k=448) in 583.86s
    v0.1.35: built tiered IVF (k=448) in 591.03s
    v0.1.35: built tiered IVF (k=448) in 591.18s

| 0.1.34 nprobe | recall | 0.1.34 latency | 0.1.35 latency at that recall (interpolated) | 0.1.35/0.1.34 at matched recall |
| --- | --- | --- | --- | --- |
| 1 | 0.1087 | 40.9 | 42.4 | 1.036 |
| 2 | 0.1898 | 75.8 | 77.6 | 1.024 |
| 4 | 0.3134 | 141.9 | 148.0 | 1.044 |
| 8 | 0.4885 | 275.1 | 286.9 | 1.043 |
| 16 | 0.6921 | 521.5 | 553.3 | 1.061 |
| 32 | 0.8309 | 949.5 | 977.7 | 1.030 |
| 64 | 0.9038 | 1852.8 | 1796.2 | 0.969 |

### Tiered IVF (BuoyIndex) synthetic mixture-of-Gaussians, dim=768 n=100,000, recall@10 vs median latency (us), both rounds
| nprobe | recall 0.1.34 | recall 0.1.35 | Δrecall | latency 0.1.34 | latency 0.1.35 | 0.1.35/0.1.34 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0506 | 0.0348 | -0.0158 | 107.2 | 80.6 | 0.752 |
| 2 | 0.0912 | 0.0694 | -0.0218 | 198.6 | 148.1 | 0.746 |
| 4 | 0.1564 | 0.1130 | -0.0434 | 377.4 | 295.8 | 0.784 |
| 8 | 0.2518 | 0.1940 | -0.0578 | 691.8 | 548.9 | 0.793 |
| 16 | 0.3370 | 0.3102 | -0.0268 | 1126.2 | 1058.0 | 0.939 |
| 32 | 0.4278 | 0.4376 | +0.0098 | 1966.4 | 2080.2 | 1.058 |
| 64 | 0.5702 | 0.5830 | +0.0128 | 3705.1 | 3917.0 | 1.057 |
    v0.1.34: built tiered IVF (k=317) in 369.89s
    v0.1.34: built tiered IVF (k=317) in 377.61s
    v0.1.35: built tiered IVF (k=317) in 326.85s
    v0.1.35: built tiered IVF (k=317) in 327.50s

| 0.1.34 nprobe | recall | 0.1.34 latency | 0.1.35 latency at that recall (interpolated) | 0.1.35/0.1.34 at matched recall |
| --- | --- | --- | --- | --- |
| 1 | 0.0506 | 107.2 | 111.4 | 1.040 |
| 2 | 0.0912 | 198.6 | 221.9 | 1.117 |
| 4 | 0.1564 | 377.4 | 431.4 | 1.143 |
| 8 | 0.2518 | 691.8 | 802.1 | 1.159 |
| 16 | 0.3370 | 1126.2 | 1273.0 | 1.130 |
| 32 | 0.4278 | 1966.4 | 2001.5 | 1.018 |
| 64 | 0.5702 | 3705.1 | 3755.3 | 1.014 |

Reading. At any fixed nprobe up to 16, **0.1.35 reaches lower recall** (−0.012 to −0.033 at 200k×128,
−0.016 to −0.058 at 100k×768) **and lower latency** (0.89–0.93× and 0.75–0.79×): the e-means partition
is less balanced, so probing the same number of lists scans fewer vectors and finds fewer true
neighbours. At nprobe ≥ 32 the two meet. **At matched recall, 0.1.35 is 2–6% slower at 200k×128 and
2–16% slower at 100k×768** (nprobe 2–16), the opposite sign from the 10k×128 gate corpus, where recall
rose. The full tiered build was 583 → 591 s at 200k×128 and 370 → 327 s at 100k×768 — the training
speed-up of phase 3 is diluted by everything else the build does. Absolute recall on this synthetic
corpus is low by design (the benchmark's own README calls it illustrative); the comparison across
tags on the same corpus is what this section claims.

**Real data was not measured.** `scripts/realivf.sh` (RecallQpsBenchmark, fashion-mnist-784, three
`nlist` values, both tags) was started and stopped after 35 minutes without a completed row —
single-threaded 50-iteration k-means on 60k×784 for three `nlist` values, twice — to give the box to
the 100k HNSW measurement. It is the one open measurement the e-means decision would benefit from.

## Verdict

1. **The e-means IVF default is reverted in 0.1.36** (`CentroidTrainer.K_MEANS` by default,
   `E_MEANS` opt-in). It was shipped unmeasured; measured, it trades recall for scan speed and is
   2–16% slower at matched recall on this corpus; its validated regime (n ≥ 10⁵, k ≥ 256, high
   intrinsic dimension) is above where `IvfBuildParams.defaults` puts most indexes (k = ⌈√n⌉);
   FAISS itself does not default to it. Real-data confirmation (fashion-mnist, SIFT-1M at k = 4√N)
   is the open measurement, not a blocker for restoring the previous default.
2. **No HNSW change.** The dim-128 "regression" did not survive fifteen forks per arm (0.917× latency
   at 10k×128, i.e. 0.1.35 faster; no reverted commit moves it). The 100k×128 raw shape is being
   re-measured at fifteen forks (`scripts/power100k.sh`) and its table is appended to this directory
   when it lands; nothing in 0.1.36 depends on it.
3. **0.1.36 also carries**: the FAISS MIT notice the e-means port shipped without; a build-time fix
   so a persistent recipe collection can be created at a path that does not exist yet (0.1.34
   regression found by `integrallis/memory`); an uncacheable `recallGate`; a comparison script that
   runs; the 0.1.35 CHANGELOG section that was missing, with these measurements.
4. **Process findings** recorded for the next release: perf commits need a measured before/after
   against the released tag (the comparison script now defaults to five forks because two cannot
   resolve a 2–5% question); a default change is a user-facing change and needs a measurement on the
   workload it changes, at the scales users run; a release gate must execute, not be restored from
   cache.

## Reproduce

`scripts/userdata-vec.sh` (instance bootstrap), `scripts/vecdriver.sh` (datasets, worktrees, the failed
first JMH phase, recall gate, SIFT floors), `scripts/vecjmh.sh` / `vecjmh2.sh` / `vecjmh3.sh` (JMH phases),
`scripts/stair.sh` + `confirm.sh` (bisect box). Derive with
`python3 scripts/derive-vectors.py raw/`, `python3 scripts/derive-stair.py raw/stair <repo>`,
`python3 scripts/derive-confirm.py raw/confirm`.
