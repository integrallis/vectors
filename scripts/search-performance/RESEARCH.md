# Research checked during the default-search change

Reading a source, implementing it, and qualifying an improvement are different
states. The search changes in this PR are exact implementation optimizations.
They do **not** implement Flash, FastHNSW, Slipstream, or graph reordering.

## Supplied ten-item work order

| Item | Source examination and code comparison | Implementation / measurement status |
|---|---|---|
| Flash | [Paper §3](https://arxiv.org/html/2502.18113v1) and local `HNSW-Flash` code/parameters: PCA, PQ, symmetric/asymmetric tables, adjacent neighbor codes and SIMD lookup batches form the method. A plain SQ8 node scorer does not reproduce it. | Original SQ8 experiments were rejected and preserved. Full Flash pipeline remains unimplemented and unmeasured here. |
| BP reordering | [Lucene implementation](https://github.com/apache/lucene/blob/main/lucene/misc/src/java/org/apache/lucene/misc/index/BpVectorReorderer.java) recursively partitions using vector centroids and maintains both ordinal maps. It has iterative partitioning cost; rewriting existing files does not make partitioning free. | Not implemented. Our FLOAT32 vector file has fixed-width rows: reordering alone cannot reduce its uncompressed byte count. Locality, remapping correctness and compaction cost need separate measurements. |
| extendCandidates | Existing retained multi-dataset results: improves Wikipedia recall but costs 1.88× construction; pairing it with reduced forward degree loses 0.029 recall. | Previously evaluated; no default promotion. No new result claimed. |
| Relative NN-Descent | [Paper §4](https://arxiv.org/html/2310.20419v1): update neighbor lists, replace a pruned edge with an edge between candidates, add reverse edges, and iterate using old/new flags. | Requires an iterative graph builder, rather than changing one insertion rule. Not implemented or newly benchmarked. |
| FastKCNA | [Paper §5.3–5.4](https://arxiv.org/html/2410.01231v2), local `FastKCNA/code`: assign levels globally, construct/refine each layer, reuse old/new relationships between iterations. | The work order's “direct drop-in” characterization is incorrect. Our streaming builder does not have those global iterations. No implementation or speedup claim. |
| OPT-SNG | [Paper §5](https://arxiv.org/html/2509.15531v1) optimizes a truncation parameter for SNG and includes the cost of repeated parameter selection in its comparison. | Current production M/efC are fixed, not selected by an internal sweep. Changing M requires a new recall qualification; no default change or transferred speedup claim. |
| Slipstream | [Paper §3](https://arxiv.org/html/2606.02992v1) retains upper-layer routing, reuses layer-zero candidates, gates reuse using displacement/local-neighborhood radius, and adapts the reduced construction beam. | Corpus order alone does not establish the needed locality. Concurrent interleavings also differ from the paper's sequential stream. Not implemented; no reduced beam or recall allowance introduced. |
| IGTM/CGTM, FGIM | [IGTM/CGTM §4](https://arxiv.org/html/2505.16064v1) merge independently built graphs; [FGIM §3–4](https://arxiv.org/html/2603.21710v1) converts existing proximity graphs into candidate k-NN graphs, refines them, and reconstructs proximity graphs. | Our `HnswGraphMerger` takes **one graph and its surviving vectors**, remapping after deletion. It is not currently the multi-graph merge operation these papers accelerate. Neither algorithm implemented or measured. |
| MN-RU | [Paper §IV](https://arxiv.org/html/2407.07871v1): replacement-update neighbor maintenance plus an auxiliary index for unfound points and dual-index querying. | Existing hierarchy-aware diagnostics and counterexample remain. A layer-zero reachability count is not a hierarchical recall ceiling. No new repair algorithm implemented. |
| Shipped-default million-scale baseline | Earlier campaign measured M=16/efConstruction=200 and retained its raw evidence. | Completed previously; the new search comparison uses a frozen graph and does not substitute for that construction baseline. |

## Additional flat-index research and implemented work

[Faiss implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes)
describe separating reusable norms from pairwise work and exploiting matrix/batch
computation. We applied the reusable-query-work principle specifically to COSINE:
the query norm is prepared once per mapped scan or per query in a heap batch,
while dot products and row norms keep
the existing accumulation structure. We did not replace direct squared distances
with `norm(q)+norm(x)-2*dot(q,x)`, whose cancellation behavior would need different
numerical tests. There is no claim that this implements Faiss's BLAS path.

In HNSW, inspection found that search results already leave a min-heap in sorted
order, but were reinserted through a duplicate-checking sorted container. That
adds quadratic duplicate scans and used two arrays sized to the entire graph.
Direct draining removes this repeated work and preserves the existing tie order.
The exact drain had already been qualified for serial construction; this change
extends it to the reusable search destination and filtered searches.

The broader literature map also contains partitioned construction, alternative
pruning rules, deletion frameworks, distributed/GPU methods, and scaling theory.
Those are not claimed as investigated implementations or completed benchmarks by
this PR. No state-of-the-art comparison has been established.

## Follow-up: exact reuse and public API integration

[PR #88's experiments](../exact-reuse/README.md) extend norm reuse to stored rows
within a heap COSINE batch, extend exact heap draining to concurrent construction,
and omit duplicate scans only inside diversity selection where IDs are already
distinct. All three are implementation optimizations with unchanged scoring and
search/build budgets; they do not implement the larger paper algorithms above.

A worker-local exact symmetric pruning cache was also implemented and measured.
It reduced vector reads but made both published construction workloads slower,
so it was removed. This distinguishes reuse that reduces counted operations from
reuse that actually reduces elapsed time.

The public `VectorCollection.searchBatch` previously bypassed the flat adapter's
batch method. The follow-up adds automatic grouping for compatible in-memory
COSINE requests, tests the public path, and measures it separately. Adapter-only
batch timings from PR #87 must not be presented as public collection API gains.
See [the qualification report](../exact-reuse/RESULTS.md) for the measured scopes,
recall deltas, controls, rejected candidates and remaining limitations.
