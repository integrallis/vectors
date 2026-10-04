# Query performance results — 2026-10-04

Compared merged main `c43c3160232dd20cf4b2ebd67f44b68c83edf56c` with
runtime `e4b4706edb05f96b077a9092a737a26b32a58892`, on one dedicated Hetzner
CCX33 (AMD EPYC Milan, 8 logical CPUs / 4 physical cores, 32 GiB RAM), Temurin
25.0.3+9. Three alternating fresh JVM pairs; median of seven samples per fork,
then median of the three forks. Lower time is better.

## Qualified changes

| Public API workload | Baseline µs/query | Candidate µs/query | Less time |
|---|---:|---:|---:|
| HNSW persisted GLOVE, ef=100 | 179.64 | 160.40 | 10.71% |
| HNSW persisted DBpedia, ef=100 | 633.33 | 518.00 | 18.21% |
| Flat in-memory COSINE GLOVE, single | 9783.47 | 1182.49 | 87.91% |
| Flat in-memory COSINE DBpedia, single | 18397.79 | 5758.12 | 68.70% |

These changes are automatic. HNSW avoids per-neighbor segment-view allocations
and reuses the batch query norm. Flat distributes large isolated COSINE scans
across spare cores, with one global expansion permit and automatic batch/burst
coordination. Flat adds one temporary `4 * rowCount` byte score buffer; this is
latency improvement using more cores, not a demonstrated reduction in CPU work.

All 56 primary workloads produced identical result-ID and raw float-score-bit
digests in all samples and forks. Graphs, query budgets, metric arithmetic and
precision are unchanged. GLOVE/Fashion queries are held out; DBpedia queries are
perturbed source vectors, so that corpus is not held-out recall evidence.

Both targeted COSINE corpora exceed the predeclared 5% improvement threshold.
The worst primary control is Fashion flat independent Q16: 4.39% more time,
within the predeclared 5% control limit. Heap HNSW has no claimed improvement;
GLOVE heap ef32 takes 3.29% more time. Euclidean/DOT flat expansion was rejected
and removed. No claim is made for persisted flat, cold pages, different hardware,
or a ranking against other libraries.

## All primary measurements

Positive percentages mean less time; negative percentages mean more time.
`mapped` is a native contiguous-segment diagnostic. `public` reopens an actual
persisted collection. HNSW is pinned to CPU 2 with JVM processor count held at 8;
flat uses all 8 CPUs. Flat batch and independent-query times are total batch
wall time divided by query count, not individual-request latency.

| Workload | Baseline µs/query | Candidate µs/query | Less time |
|---|---:|---:|---:|
| glove-hnsw-heap-ef32 | 38.83 | 40.11 | -3.29% |
| glove-hnsw-heap-ef100 | 105.18 | 106.61 | -1.36% |
| glove-hnsw-heap-ef128 | 133.59 | 135.46 | -1.40% |
| glove-hnsw-heap-ef512 | 460.03 | 459.59 | +0.10% |
| glove-hnsw-heap-filtered128 | 748.32 | 751.23 | -0.39% |
| glove-hnsw-mapped-ef32 | 57.51 | 44.44 | +22.72% |
| glove-hnsw-mapped-ef100 | 146.19 | 121.52 | +16.88% |
| glove-hnsw-mapped-ef128 | 182.59 | 153.18 | +16.11% |
| glove-hnsw-mapped-ef512 | 603.04 | 507.06 | +15.92% |
| glove-hnsw-mapped-filtered128 | 821.48 | 791.13 | +3.69% |
| glove-hnsw-public-ef32 | 74.76 | 68.43 | +8.47% |
| glove-hnsw-public-ef100 | 179.64 | 160.40 | +10.71% |
| glove-hnsw-public-ef128 | 228.07 | 205.18 | +10.04% |
| glove-hnsw-public-ef512 | 710.66 | 647.99 | +8.82% |
| dbpedia-hnsw-heap-ef32 | 147.83 | 145.24 | +1.75% |
| dbpedia-hnsw-heap-ef100 | 385.86 | 374.97 | +2.82% |
| dbpedia-hnsw-heap-ef128 | 474.90 | 468.04 | +1.44% |
| dbpedia-hnsw-heap-ef512 | 1531.71 | 1539.95 | -0.54% |
| dbpedia-hnsw-heap-filtered128 | 1980.09 | 1990.87 | -0.54% |
| dbpedia-hnsw-mapped-ef32 | 168.82 | 150.60 | +10.80% |
| dbpedia-hnsw-mapped-ef100 | 418.87 | 389.26 | +7.07% |
| dbpedia-hnsw-mapped-ef128 | 596.81 | 475.83 | +20.27% |
| dbpedia-hnsw-mapped-ef512 | 1670.16 | 1546.71 | +7.39% |
| dbpedia-hnsw-mapped-filtered128 | 2125.12 | 2025.90 | +4.67% |
| dbpedia-hnsw-public-ef32 | 244.37 | 222.20 | +9.07% |
| dbpedia-hnsw-public-ef100 | 633.33 | 518.00 | +18.21% |
| dbpedia-hnsw-public-ef128 | 700.10 | 658.55 | +5.93% |
| dbpedia-hnsw-public-ef512 | 2109.43 | 2025.34 | +3.99% |
| fashion-hnsw-heap-ef32 | 75.82 | 76.81 | -1.31% |
| fashion-hnsw-heap-ef100 | 168.94 | 166.27 | +1.58% |
| fashion-hnsw-heap-ef128 | 203.74 | 199.48 | +2.09% |
| fashion-hnsw-heap-ef512 | 564.94 | 559.29 | +1.00% |
| fashion-hnsw-heap-filtered128 | 730.34 | 730.70 | -0.05% |
| fashion-hnsw-mapped-ef32 | 92.45 | 92.06 | +0.42% |
| fashion-hnsw-mapped-ef100 | 217.73 | 210.67 | +3.24% |
| fashion-hnsw-mapped-ef128 | 256.57 | 250.41 | +2.40% |
| fashion-hnsw-mapped-ef512 | 622.64 | 624.74 | -0.34% |
| fashion-hnsw-mapped-filtered128 | 738.93 | 746.16 | -0.98% |
| glove-flat-public-single | 9783.47 | 1182.49 | +87.91% |
| glove-flat-public-batch4 | 1225.62 | 1226.72 | -0.09% |
| glove-flat-public-batch16 | 1187.50 | 699.43 | +41.10% |
| glove-flat-public-batch32 | 524.08 | 514.65 | +1.80% |
| glove-flat-independent4 | 1191.02 | 1233.65 | -3.58% |
| glove-flat-independent16 | 894.66 | 873.93 | +2.32% |
| dbpedia-flat-public-single | 18397.79 | 5758.12 | +68.70% |
| dbpedia-flat-public-batch4 | 5350.93 | 5309.88 | +0.77% |
| dbpedia-flat-public-batch16 | 3526.29 | 3470.48 | +1.58% |
| dbpedia-flat-public-batch32 | 1421.68 | 1414.16 | +0.53% |
| dbpedia-flat-independent4 | 5322.32 | 5324.29 | -0.04% |
| dbpedia-flat-independent16 | 4227.43 | 4243.36 | -0.38% |
| fashion-flat-public-single | 10109.45 | 10282.26 | -1.71% |
| fashion-flat-public-batch4 | 2987.01 | 3029.24 | -1.41% |
| fashion-flat-public-batch16 | 1814.26 | 1812.75 | +0.08% |
| fashion-flat-public-batch32 | 1816.77 | 1807.68 | +0.50% |
| fashion-flat-independent4 | 2934.98 | 2986.79 | -1.77% |
| fashion-flat-independent16 | 2706.32 | 2825.03 | -4.39% |

## Warmup investigation

The original GLOVE heap filtered workload creates a fresh searcher per query.
Both arms show a step down during the seven measured samples (roughly 800 to
650 µs/query). A JIT transition is a possible explanation, not proven by these
logs. Its aggregate is 0.39% more time, but that timing instability warrants a
separate check. The diagnostic `heap-filtered` mode retains fresh construction,
warms for at least 128 passes (32,768 queries), and separately measures a reused
searcher. Original samples remain in the evidence; the diagnostic does not
replace the primary table. Across three additional alternating pairs, fresh
searchers took 649.52 → 646.01 µs/query (0.54% less time), and reused searchers
599.15 → 599.74 µs/query (0.10% more time). Both retain identical output digests.

## Allocation and validation

Calling-thread allocation at ef100 fell from 96,213 to 5,392 bytes/query for
public GLOVE and from 116,549 to 32,450 for public DBpedia. These serial HNSW
measurements include result materialization. Flat worker allocations are not
included by this probe and are not presented as total allocation.

Rebuilding `1d8797d` after the CI fixes produced all eight runtime/dependency jars
byte-identical to measured `e4b4706`; see
[evidence/qualification/final-runtime-identity.json](evidence/qualification/final-runtime-identity.json).
The later changes are documentation, benchmark diagnostics and retained evidence.
VPS `spotlessJavaCheck`, `spotbugsMain`, five `ParallelFlatScanTest` tests and four
`FlatCollectionBatchTest` tests passed together after those fixes. Full CI remains
a required merge gate, including coverage, recall, integration and portability.
The earlier interrupted VPS full DB suite is explicitly retained as interrupted,
not counted as a pass.

## Evidence and reproduction

See [README](README.md) for inputs, commands, test history and rejected experiments.
The `evidence/` directories retain every raw sample, exact JVM commands, source,
jar/input SHA-256 provenance, summaries and independent audit output. Summary
JSON includes each fork median, so the aggregate is not the only visible result.

Run the independent audit for each primary directory:

```sh
python3 scripts/query-performance/audit.py scripts/query-performance/evidence/flat-v8-final
python3 scripts/query-performance/audit.py scripts/query-performance/evidence/hnsw-v8-final
```

The fixture/input archive is required for replay; logs do not reconstruct data.
The GLOVE public fixture was built once using baseline jars, then copied without
rebuilding its graph for both arms. Original input archive hashes and the final
runtime-identity record connect the measurements to the submitted code.

The supplemental local replay archive `replay-assets.tar.gz` contains the frozen
candidate jars for all attempts and the baseline-built public GLOVE fixture:
18,546,491 bytes, SHA-256
`eec753e9a2f42091e0707a4c3f5d5c234b74902098a58081dd76fc029a56aa67`.
It is retained in the workspace research directory alongside the original input
archive; it is not downloaded automatically by the runner.
