# VPS qualification: exact flat batches and concurrent HNSW construction

Baseline: merged main `212ad282d719a985706b8024ad310c73e5e85430` (PR #87).
Final production source: `7df8519`. This comparison measures the increment after
PR #87; percentages must not be added to earlier measurements.

Host: dedicated Hetzner CCX33, AMD EPYC Milan, 8 logical / 4 physical processors,
32 GiB RAM, Temurin 25.0.3+9, Linux x86-64. Timings ran serially on the VPS.
Each flat case has three alternating fresh-JVM pairs, eight warmups and seven
measured repetitions per process. Results below use median-of-process-medians;
they are workload observations, not a confidence interval or universal guarantee.

## Qualified defaults

Compared with the previous merged main, on this VPS:

| Workload | Median time reduction |
|---|---:|
| Public in-memory COSINE flat batch, GLOVE 100k × 100, Q64 | **27.99%** |
| Public in-memory COSINE flat batch, DBpedia 100k × 512, Q64 | **56.94%** |
| Concurrent HNSW construction, 20k rows, 8 workers | **8.28–8.91%** |
| Persistent DBpedia ingestion, 100k × 512, four commits | **3.47%** |

All flat output digests matched. Single-worker HNSW graph/query digests matched.
Parallel HNSW's worst held-out recall delta was -0.195 percentage points; the
persistent replay's worst in-corpus delta was -0.06 points. Both passed the
unchanged -0.5-point gate. No runtime option, lower M/ef, approximate scoring or
weaker gate is required. This is not a claim of identical parallel-build recall.

The final public comparison settles the corpus before warmup. Its worst control
was 3.16% slower; earlier unsettled runs showed much larger variability and are
fully retained below. These are steady-state workload results, not a universal
speedup or a claim about startup/GC-transition latency.

## HNSW construction

20,000-row prefixes, M=16, efConstruction=200, k=10, seeds 17/42/91, unchanged
search ef=32/128/512. Official held-out query files supply 256 queries per corpus;
exact neighbors are recomputed over the selected prefix. The HNSW module in the
final runtime is identical to the isolated combined arm.

| Corpus | Workers | Baseline median | Candidate median | Less build time |
|---|---:|---:|---:|---:|
| fashion-mnist-784-euclidean | 1 | 11.288 s | 10.998 s | 2.56% |
| fashion-mnist-784-euclidean | 8 | 2.665 s | 2.427 s | 8.91% |
| glove-100-angular | 1 | 12.324 s | 12.018 s | 2.48% |
| glove-100-angular | 8 | 2.618 s | 2.401 s | 8.28% |

All six single-worker encoded graphs and query digests matched exactly. Parallel
construction changes scheduling and produced different graphs. Every paired
recall gate passed the existing -0.005 threshold; the worst observed delta was
-0.00195313 (**-0.195 percentage points**) for GLOVE at ef=128. This is not a claim
of identical parallel-build recall. No M, efConstruction, search ef, precision,
or recall gate was reduced.

Query latency is not the targeted gain. Across the twelve corpus/worker/ef
comparisons it ranged from 2.80% more time to 2.23% less time. For eight-worker
builds specifically, it ranged from 1.13% more to 1.83% less. Raw per-seed values
are in [combined records](evidence/combined-results/records.json).

The direct heap drain reduced the fixed 4,096-result allocation probe from
65,624 to 32,824 bytes. The distinct insertion bypass is package-private and only
used where candidate uniqueness and disjoint accepted/blocked sets establish its
precondition; public insertion still checks duplicates.

## Flat adapter (separate from public collection API)

The final adapter arm uses both-norm reuse only for batches of at least four
queries and a compatible SIMD width. The old smaller-batch scan remains intact.
All 36 case digests matched exactly across baseline and candidate.

| COSINE corpus | Queries | Baseline median | Candidate median | Less batch time |
|---|---:|---:|---:|---:|
| GLOVE 100k × 100 | 4 | 10.35 ms | 9.10 ms | 12.12% |
| GLOVE 100k × 100 | 16 | 33.36 ms | 25.87 ms | 22.44% |
| GLOVE 100k × 100 | 64 | 126.26 ms | 96.98 ms | 23.19% |
| DBpedia 100k × 512 | 4 | 42.33 ms | 37.94 ms | 10.39% |
| DBpedia 100k × 512 | 16 | 100.61 ms | 83.49 ms | 17.02% |
| DBpedia 100k × 512 | 64 | 356.03 ms | 291.50 ms | 18.12% |

Controls matter: small COSINE batches ranged from 1.79% more time to 5.54% less;
repeated individual COSINE searches ranged from **7.50% more time** to 3.90% less.
The EUCLIDEAN controls ranged from 0.88% more time to 2.77% less. Individual search
source was unchanged, but these observed differences are retained and not
silently relabeled as improvements. See [all flat cases](evidence/flat-v3-final/summary.json).

## Rejected and isolated candidates

- The first both-norm flat implementation regressed GLOVE by roughly 24–27%; it
  was replaced with a direct SIMD dot loop and dispatch outside the scan.
- The second flat version regressed GLOVE Q=2 by 6.81%; final routing retains the
  old scan below Q=4 automatically.
- Worker-local symmetric HNSW pair caching reduced a fixed probe's vector reads
  by 24.4% but increased eight-worker build time by 13.1% on GLOVE and 7.9% on
  Fashion. It was removed; reduced reads did not establish a speedup.
- The direct drain alone improved concurrent build time by 4.1% / 7.6% but its
  persistent replay was 0.57% slower. Distinct insertion alone improved build
  time by 3.1% / 0.3%. The combination reached 8.3% / 8.9% and improved the first
  persistent replay by 2.59%. Final public-API runtime replay is recorded below.

All rejected-arm commands, hashes, raw samples and results are retained in
[evidence](evidence). The larger supplied research algorithms remain explicitly
unimplemented in [the research examination](../search-performance/RESEARCH.md).
No state-of-the-art ranking has been established.

## Public collection API: initial eight-pass warmup

Three pairs, 100k GLOVE/DBpedia rows, 60k Fashion control rows, query counts
1/4/16/32/64, default field projections and k=10. All 30 output digests matched
between revisions. Both Q64 COSINE cases passed the predeclared 5% gate.

| COSINE corpus | Queries | Baseline median | Candidate median | Less public batch time |
|---|---:|---:|---:|---:|
| GLOVE 100k × 100 | 32 | 17.67 ms | 14.21 ms | 19.56% |
| GLOVE 100k × 100 | 64 | 33.82 ms | 24.36 ms | 27.96% |
| DBpedia 100k × 512 | 32 | 78.38 ms | 41.89 ms | 46.56% |
| DBpedia 100k × 512 | 64 | 170.02 ms | 76.64 ms | 54.92% |

These timings include the public API's dispatch, scoring and projection work.
On this eight-processor host, Q32/Q64 qualify automatically; smaller batches,
EUCLIDEAN, normalized, cache-enabled, filtered, mixed-k and persistent flat
collections retain their prior route. No runtime option is required.

The controls were variable: DBpedia Q1 batch was **16.43% slower**, Fashion Q4
batch **16.15% slower**, and Fashion Q64 batch **5.48% slower**. Meanwhile the
unchanged GLOVE single-query path appeared 34.53% faster, with baseline JVM medians
ranging from 2.98 to 4.66 ms. This run cannot establish that those untouched paths
improved. All measurements are retained in [the initial public summary](evidence/public-api-final/summary.json).
A separately predeclared longer-warmup investigation follows below; it uses the
same production jars, inputs and complete set of cases.

## Final runtime: persistent collection ingestion

The unchanged `DefaultCollectionReplayBenchmark` replays 100,000 × 512 DBpedia
vectors through four ordinary commits of 25,000 documents, with eight HNSW workers.
Three alternating fresh-JVM pairs measured the entire ingestion path, then reopened
saved collections with the same baseline query runtime for recall. The initial
construction level seed is time-generated in this existing benchmark, so exact
parallel graph reconstruction is not promised. Saved graph artifacts are retained.

| Pair | Baseline ingest | Candidate ingest |
|---|---:|---:|
| 1 | 40.922 s | 39.469 s |
| 2 | 40.889 s | 39.625 s |
| 3 | 40.325 s | 39.090 s |

Median ingestion time fell from **40.889 s to 39.469 s (3.47%)**. This modest end-to-end
gain is distinct from the 8.3–8.9% graph-only build gain. The earlier combined
runtime replay measured 2.59% less time; both comparisons are retained.

Recall uses 1,000 evenly spaced **in-corpus** queries, exact top-10 over the source
prefix, ef=10/16/32/64/128/256. Every unchanged -0.005 gate passed. Worst observed
delta was **-0.0006 (-0.06 percentage points)** at ef=10. Source hashes remained
unchanged. See [all replay deltas](evidence/final-replay/summary.json).

## Regression validation

The full VPS database suite passed **910 tests, zero skipped/failed**, including
commit-memory, vector ownership, generation lifecycle, concurrent readers,
normalization, persistence, tombstones, filters and the new public batch path.
The database coverage gate passed. JFR recorded eight workers for 32 public batch
queries versus 32 on the old routing. Core/HNSW tests and coverage passed; the
production commit passed all 13 CI checks, including ARM64, 128-bit x86 SIMD and
scalar portability. Performance qualification is specific to the dedicated x86
host; those portability checks are not cross-architecture speed measurements.

## Warmup and heap-layout investigation

The next comparison added at least three seconds of warmup per case, keeping all
30 cases and the same jars. Q64 public batches again improved (28.08% GLOVE,
53.19% DBpedia), with all digests equal. Most controls narrowed, but GLOVE's
repeated single Q64 searches became **124.75% slower**: baseline fork medians
177.83/178.03/181.00 ms versus candidate 433.83/400.13/399.59 ms. This anomaly
prevented promotion at that point. See [the complete run](evidence/public-api-warm3/summary.json).

A subsequent diagnostic run enabled JFR, compiler logging and GC logging. The
**baseline also reproduced** roughly 429–461 ms Q64 single passes; candidate
showed roughly 398–434 ms. Both recorded a G1 young collection around 20 seconds,
relocating the heap-backed corpus before the later cases. These diagnostic timings
include instrumentation and are not performance qualification results. They point
to heap placement as a confounder, rather than establishing a scorer regression.
Raw [baseline GC](evidence/glove-baseline-gc.log), [candidate GC](evidence/glove-transition-gc.log)
and corresponding timing logs are retained; full JFR/compiler traces are in the
local VPS archive.

The next predeclared comparison calls `System.gc()` once after each benchmark
collection is built, before warmup or measurement, for both revisions. This settles
retained rows into old generation before query allocations can relocate them at
different points during a fork. Production code and jars remain unchanged. The
same three-second/eight-pass warmup minimum, all query counts, seven measurements
and three alternating pairs remain. This qualifies a **steady-state heap** workload;
it does not establish startup or GC-transition latency. No GC call or warmup is
added to the library, and callers need no option to use the performance changes.

## Final public API comparison: identical corpus placement

The completed three-pair comparison passes every output digest and both Q64
speed gates. The large single-search anomaly disappears: GLOVE single Q64 is
174.864 ms baseline versus 174.776 ms candidate (0.05% less time).

| COSINE corpus | Queries | Baseline median | Candidate median | Less public batch time |
|---|---:|---:|---:|---:|
| GLOVE 100k × 100 | 32 | 16.658 ms | 13.323 ms | 20.02% |
| GLOVE 100k × 100 | 64 | 33.035 ms | 23.789 ms | 27.99% |
| DBpedia 100k × 512 | 32 | 80.352 ms | 38.562 ms | 52.01% |
| DBpedia 100k × 512 | 64 | 165.811 ms | 71.392 ms | 56.94% |

All repeated individual-search controls ranged from 3.16% more time to 1.69%
less. Ineligible public batch controls ranged from 2.41% more to 34.04% less;
the large apparent GLOVE Q4 improvement is on the old route and is not attributed
to this optimization. Euclidean batch controls ranged from 1.34% more to 2.10%
less. [Every case and fork median](evidence/public-api-settled/summary.json) is
retained, including regressions. The baseline/candidate jars match the earlier
runs exactly; only benchmark preparation changed.

`audit-flat.py` independently recomputed the raw measurements and all digest
sets for the adapter comparison and all three public-API comparisons. Its
negative checks reject altered summaries, altered digests and duplicate sample
numbers. The audit establishes internal consistency of the retained measurements;
it does not turn three JVM pairs into a universal performance guarantee.
