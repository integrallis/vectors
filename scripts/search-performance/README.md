# Default search performance

The change prepares the COSINE query norm once per flat scan (heap batch
and mapped storage) and drains HNSW result heaps directly into reusable sorted
storage. HNSW retains the same traversal, scores, tie order, and search budget.
Each searcher loses an `int[N]` and a `float[N]` (8N bytes plus array headers).
Single-query heap scans retain their original kernel. The flat implementation
preserves the existing four-accumulator arithmetic;
short vectors and scalar providers retain their existing cosine path.

## Reproduce

Use JDK 25 and two checkouts. The baseline for this change is merged main
`fc7c525e95376ceb20729e4f0bb4364b4e25a82f`.

```sh
python3 scripts/search-performance/run.py /absolute/baseline /absolute/candidate \
  /absolute/new-results-directory --rounds 3
```

The output directory must be new. Both revisions are built before measurement.
All runtime jars are copied into the output directory, and one identical harness
is compiled against the baseline API. The runner alternates fresh baseline and
candidate JVMs, without concurrent builds or tests. It records source revisions,
patches, Java/platform details, commands, raw samples and a summary.

Workloads: 10,000 synthetic COSINE rows at 128/512/768 dimensions, 64 held-out
queries, k=10, heap single/batch and mapped single searches. HNSW uses 20,000
normalized 128-dimensional rows, 1,000 held-out queries, DOT_PRODUCT, M=16,
efConstruction=200, and efSearch=32/128/512, unfiltered and 50%-selective filtered.
The baseline builds one graph, serializes it, and **every arm reads that exact
file**, so graph construction scheduling cannot change the search comparison.
Each workload warms first, then records seven passes per JVM. Summaries take the
median within each fork and then across forks. Hashes cover every returned ID
and score bit; mismatches fail the run. Searcher allocation is measured separately.

These are warm-cache search workloads. Run them on an idle dedicated benchmark host. They do not establish large-corpus,
cold-storage, construction, embedding-model, or competitor/SOTA performance.

## Regression checks

```sh
./gradlew :vectors-core:test --tests '*PreparedCosineTest' --tests '*CosineAllocationTest' \
  :vectors-hnsw:test :vectors-db:test --tests '*PreparedFlatScanTest' \
  --tests '*MappedFlatScanAdapterTest*' --tests '*WriterGraphLifecycleTest' \
  --tests '*SuccessorVectorOwnershipTest' --max-workers=2
```

The initial missing-method failures and the allocation assertion's failure on
main are retained in `evidence/`. The compiled cosine test leaves query norm
preparation cold while warming the pair-scoring methods, matching the difference
in call frequency in actual flat scans. It also checks scalar tails, special
values and segment scoring. CI runs core/HNSW contracts on ARM64, x86 with 128-bit
SIMD, and scalar fallback.

## Failed measurements retained

- First harness attempt used unnormalized vectors with DOT_PRODUCT in HNSW;
  NodeQueue rejected negative scores. No HNSW timing claim uses that run.
- Second attempt exposed a cold query-norm reduction versus compiled cosine
  reduction mismatch at dimension 128. The run was stopped. The norm preparation
  now uses the existing explicit reduction tree rather than a JIT-tier-dependent
  `reduceLanes(ADD)` result.

- Laptop latency measurements are withdrawn: workstation contention made them
  unsuitable for qualification. They are not evidence of a speedup or slowdown.
  The single-query heap experiment was removed; that path retains the baseline
  kernel. Batch cosine dispatch is outside the inner loop, preserving other metrics.

## Published datasets

`PublishedSearchPerformance.java` also accepts little-endian fbin files: two int32
header fields (row count, dimension), followed by row-major float32 values.
Use the first 100,000 GloVe-100 train rows and all 60,000 Fashion-MNIST train rows
from ANN-Benchmarks, preserving source order, plus their first 256 official test
queries. Name inputs `<dataset>-train.fbin` and `<dataset>-test.fbin`, with dataset
names `glove-100-angular` and `fashion-mnist-784-euclidean`.

Compile with both companion Java harnesses against the frozen baseline classpath.
Run `PublishedSearchPerformance INPUT_DIRECTORY SHARED_FIXTURE_DIRECTORY` with
baseline first, then alternate baseline/candidate JVMs. Both arms must share the
fixture directory: the baseline saves the graph and exact top-10 oracle there.
Flat timing uses the first 32 official queries over the full selected corpus. HNSW
uses all 256 queries against the first 20,000 train rows, M=16, efConstruction=200,
seed=421, efSearch=32/128/512. Report recall against the selected graph corpus,
not the original full-dataset neighbor file. Check complete ID/score-bit hashes
and recall counts across arms. Retain input hashes, commands and all raw samples.

## Persistent corpus and other metrics

After the first campaign completes, the same frozen jars can run against an
existing, tombstone-free FLOAT32 HNSW generation:

```sh
python3 scripts/search-performance/run-frozen-collection.py /absolute/first-results \
  /absolute/generation-directory /absolute/new-collection-results --rounds 3
```

This records input-file SHA-256 hashes and uses 32 deterministically perturbed
source vectors as queries. It compares heap and mapped flat COSINE, mapped HNSW
at ef=32/128/512, and heap single/batch EUCLIDEAN and DOT_PRODUCT controls. This
is a search replay, not held-out recall estimation. Result digests must match
between revisions. Runtime provenance comes from the first campaign's frozen
jars, even if documentation or benchmark sources are subsequently updated.

See [the source examination](RESEARCH.md) for what was checked against the supplied
research and what these changes do and do not implement. Measured tables and raw
fork medians are recorded in PR #87; the runner also writes them to `summary.json`.

Final dedicated-VPS measurements and all raw timing logs: [2026-10-03 evidence](evidence/vps-20261003/README.md).
