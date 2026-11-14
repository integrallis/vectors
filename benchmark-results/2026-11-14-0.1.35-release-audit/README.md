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

## HNSW search regression — bisect

_pending: `derived/staircase.md` (25-point first-parent staircase over the commits touching
`vectors-hnsw`/`vectors-db` main sources, `scripts/stair.sh`) and `derived/confirm.md`
(`v0.1.34` / `v0.1.35` / `fixA` / `fixB` on `searchHnsw` and `buildHnsw`, `scripts/confirm.sh`)._

## Phase 2 (facade and IVF search at 10k/100k × 128/768) and phase 3 (k-means vs e-means training cost)

_pending: `scripts/vecjmh2.sh`, `scripts/vecjmh3.sh`; derived into `derived/all.md`._

## Verdict

_pending._

## Reproduce

`scripts/userdata-vec.sh` (instance bootstrap), `scripts/vecdriver.sh` (datasets, worktrees, the failed
first JMH phase, recall gate, SIFT floors), `scripts/vecjmh.sh` / `vecjmh2.sh` / `vecjmh3.sh` (JMH phases),
`scripts/stair.sh` + `confirm.sh` (bisect box). Derive with
`python3 scripts/derive-vectors.py raw/`, `python3 scripts/derive-stair.py raw/stair <repo>`,
`python3 scripts/derive-confirm.py raw/confirm`.
