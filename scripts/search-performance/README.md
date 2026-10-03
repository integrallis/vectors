# Default search performance

The change prepares the COSINE query norm once per flat scan (heap, heap batch,
and mapped storage) and drains HNSW result heaps directly into reusable sorted
storage. HNSW retains the same traversal, scores, tie order, and search budget.
Each searcher loses an `int[N]` and a `float[N]` (8N bytes plus array headers).
The flat implementation preserves the existing four-accumulator arithmetic;
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

These are local, warm-cache search workloads. They do not establish large-corpus,
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
