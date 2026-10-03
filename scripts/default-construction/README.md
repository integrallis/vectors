# Default construction performance

These changes apply automatically. Graph parameters, float precision, insertion task dispatch,
and on-disk formats are unchanged. There is no new setting.

- Cosine scoring returns its scalar result without allocating a three-float temporary array.
  A copy of the old arithmetic is checked bit-for-bit across vector widths and scalar tails,
  in interpreted and fully warmed compiled JVMs.
- Serial search-result heaps drain directly into neighbor arrays, preserving equal-score ordering.
  Parallel construction retains its original heap conversion and uncached pruning implementation.
- The serial builder caches exact diversity scores (192 KiB per builder). Full pair keys make
  collisions evictions, never false hits. Parallel construction does not use this cache.
- The serial builder protects its query with one reusable copy for shared-buffer vector sources.
- Quantizer selection uses a three-way partition so repeated training values cannot cause the
  previous quadratic partition behavior.

## Reproduce

Use JDK 25 and two separate checkouts. The starting revision is
`4bc505b3e8c0c217e9d66727d643921a0d8efd55`; it already includes the earlier batch-write fix.
Run from the candidate checkout:

```sh
python3 scripts/default-construction/run.py /path/to/baseline "$PWD" /new/output/collection \
  --generation /read-only/collection/gen-0000000000000063 \
  --count 100000 --cadence 25000 --threads 8 --queries 1000 --rounds 3 --first candidate

python3 scripts/default-construction/run.py /path/to/baseline "$PWD" /new/output/parallel \
  --corpus /data/wikipedia_base_1M.bin --query-file /data/wikipedia_query.bin \
  --count 30000 --threads 8 --queries 1000 --rounds 3

python3 scripts/default-construction/run.py /path/to/baseline "$PWD" /new/output/serial \
  --corpus /data/wikipedia_base_1M.bin --query-file /data/wikipedia_query.bin \
  --count 30000 --threads 1 --builder serial --queries 1000 --rounds 3
```

The fbin files have little-endian int32 row/dimension headers followed by float32 rows.
The collection replay requires existing cached embeddings; it never calls an embedding API.
Keep output directories outside both source checkouts. Inputs are read-only. Every replay uses a new destination and preserves the source normalization
flag, IDs, text, metadata and hashes. The driver refuses an existing output directory.
It records source revisions, dirty patches and untracked source archives, JDK/host details,
commands and raw output. Project jars are frozen before timing; one fresh JVM runs at a time.
Do not run builds or other benchmarks concurrently with timings.

All runs use M=16 and efConstruction=200. Graph-only runs include a 5,000-row warmup and report
construction time plus a digest of every node, level, edge and raw score. Collection times include
all four commits; reopening is reported separately. Collection seeds follow the ordinary
time-seeded default. Parallel insertion is nondeterministic even for fixed level seeds.

Recall is checked against exhaustive float top-10 truth for the selected prefix at efSearch
10/16/32/64/128/256. Public queries are held out; collection probes use evenly spaced stored rows
and therefore measure a different, easier workload. Reject any paired loss above 0.005 absolute
recall. Do not infer full-million-row or embedding-service throughput from these smaller replays.

## Regression checks

```sh
./gradlew spotlessCheck build :vectors-bench:recallGate complianceCheck -x :docs:build
```

CI also runs core, HNSW and quantizer contracts on ARM64, x86 with 128-bit vectors, and x86 scalar
fallback. Timing comparisons are manual, on an otherwise idle host; CI protects correctness,
recall and removal of the allocation rather than imposing noisy wall-clock thresholds.

## Rejected experiments

The earlier SQ8 construction API is excluded: it required opting in and failed recall gates.
Exact per-worker caching and mixed/native staging variants also regressed persistent throughput.
Bounded worker loops made the collection faster but exceeded the parallel recall-loss gate in one
run (0.0071 at ef=10); the original executor dispatch was restored. Those timings are not shipping
performance claims. A subsequent variant retaining parallel heap conversion and lazy pruning
also failed (0.0062 at ef=10); both optimizations were removed from the parallel path. Historical
source and raw results remain in the local campaign archive. A third parallel variant failed by
0.014 at ef=10. All concurrent-builder edits, including independent bug fixes, were removed.
The final concurrent builder and WorkContext class files are byte-identical to the baseline.
Final qualification therefore covers the affected serial build path (full graph digests and
held-out recall), plus the persistent COSINE path (timings and exhaustive recall). This scope
reduction is not a claim that the rejected parallel variants passed their gate.

Final measurements and raw evidence are in [RESULTS.md](RESULTS.md).
