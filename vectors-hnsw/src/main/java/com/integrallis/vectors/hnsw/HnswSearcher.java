/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.FusedSimilarity;
import com.integrallis.vectors.core.SimilarityFunction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/**
 * Hierarchical search over an HNSW graph. Performs greedy descent through upper layers followed by
 * beam search at layer 0.
 *
 * <p>Scoring is delegated to a {@link NodeScorerFactory}, enabling both full-precision and
 * quantized scoring to share the same search algorithm.
 *
 * <p>Not thread-safe — owns per-search scratch buffers (NodeQueues plus a version-stamped visited
 * tag array reset in O(1) per query). For concurrent queries, create a separate {@code
 * HnswSearcher} per thread via {@link HnswIndex#searcher()}.
 */
public final class HnswSearcher {

  private final HnswGraph graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;
  private final NodeScorerFactory scorerFactory;

  // Scratch buffers reused across searches — zero allocation per search.
  // Version-stamped visited set (hnswlib-style): a node is "visited" iff
  // visitedTag[id]==visitedGen.
  // Resetting between searches is an O(1) generation bump, not an O(graph.size()) BitSet.clear()
  // that
  // zeroed all ~N bits every query (a real cost at N=1.18M — most of which are never touched).
  private final int[] visitedTag;
  private int visitedGen;
  // EXPERIMENT: monotonic node-expansion counter (= object-store GETs in the co-located block
  // design).
  long nodesVisited;
  private final NodeQueue candidates;
  private final NodeQueue results;
  // Scratch buffer for rescore(): avoids allocating a new NodeQueue on every two-pass call.
  private final NodeQueue rescoreHeap;
  // Scratch arrays for beamSearch() result reversal: pre-sized to graph capacity.
  private final int[] tmpNodes;
  private final float[] tmpScores;
  // Scratch buffers for fused bulk neighbor scoring. Sized to at least the maximum layer-0 degree
  // (2*M) so a single pass over an origin's full neighbor list never overflows when using the
  // neighbor-batch path (Fused ADC).
  //
  // Why 64: matches the Fused-ADC PQ table prefetch pattern — 64 neighbors per pass keeps the
  // scoring inner loop's working set (16-32 KB depending on PQ subspace count) inside L1 across
  // every contemporary x86/ARM data-cache. Going wider (e.g. 128) pushes the working set past
  // L1 on smaller-cache parts (Apple M1 L1d = 64 KB, Zen 3 L1d = 32 KB) and we observed regress;
  // going narrower amortises less of the dispatch overhead. Vamana uses 128 because its scan
  // path is graph-only (no PQ table) and the working set is correspondingly smaller per neighbor.
  private static final int BULK_BATCH = 64;
  private final int[] bulkIds;
  private final float[] bulkScores;
  private final int bulkCapacity;

  // Scorer scratch reused across queries. This searcher is per-thread (owns its scratch), so the
  // default full-precision scorer closes over these fields instead of allocating a fresh pool per
  // query. Sized to bulkCapacity (>= BULK_BATCH) so the fused GEMV path can score a full neighbor
  // list — including greedy descent's per-node batch — in one call without overflowing the pool.
  // For typical M (<= 31) bulkCapacity == BULK_BATCH (64); it only grows past 64 to preserve this
  // bound when M is large.
  private final float[][] scorerPool;
  private final float[] scorerOut;
  // Searcher-lifetime reusable off-heap query segment for the zero-copy (segment) scoring path.
  // Allocated ONCE in the constructor and refilled per query via MemorySegment.copy — NOT a new
  // Arena/segment per query. Null unless this searcher uses the default scorer AND the underlying
  // vectors support segments.
  private final Arena scorerArena;
  private final MemorySegment queryScratchSeg;
  // Reusable per-searcher scratch of zero-copy row-segment slices for the fused segment GEMV
  // (bulkScore on the zero-copy path). Sized to bulkCapacity and refilled with vectorSegment()
  // views per bulkScore call — NEVER re-allocated per call. Null unless the segment scorer is used.
  private final MemorySegment[] scorerRows;
  // True when scorerFactory == this::defaultScorer (stateful — bound to this searcher's scratch).
  // Multi-start workers must then build their OWN default scorer rather than share this one, so a
  // worker never touches another thread's scorer scratch.
  private final boolean usesDefaultScorer;
  // Reusable single-entry-point array — avoids the per-search new int[]{ep} allocation in search /
  // searchFiltered / pickSeeds / per-worker seeding. Set [0] immediately before each call.
  private final int[] singleEntry = new int[1];
  // Reusable beam-search result array (grow-if-needed + clear per call).
  //
  // SAFETY INVARIANT: the array returned by beamSearch/beamSearchFiltered is fully consumed (by
  // extractTopK, pickSeeds, or the multi-start merge) BEFORE the next beamSearch on the SAME
  // searcher. Single-start (search/searchFiltered) consumes it immediately via extractTopK.
  // Multi-start's Phase-2 pickSeeds beamSearch is drained into the seeds int[] before any further
  // beamSearch, and the parallel workers each use a SEPARATE per-worker searcher (own scratch), so
  // per-worker results never alias. No call site violates this invariant.
  private NeighborArray reusableResult;

  /**
   * Optional SSD prefetch hook — called for each neighbor id before the scoring loop in {@link
   * #beamSearch} to submit async touch-reads. {@code null} means no prefetching (default). Set via
   * {@link #setPrefetchHook(IntConsumer)}.
   */
  private IntConsumer prefetchHook;

  /**
   * Sets (or clears) the SSD prefetch hook.
   *
   * <p>When non-null, the hook is called for every neighbor id in every candidate's neighbor list
   * <em>before</em> the neighbor scoring pass. This allows an {@link AsyncVectorPrefetcher} to
   * submit background touch-reads that bring mmap pages into the OS page cache before the main
   * thread needs them.
   *
   * <p>Thread safety: this method is only called by the thread that owns this {@code HnswSearcher}
   * (which is the thread that retrieved it from {@link HnswIndex#searcher()}).
   *
   * @param hook the prefetch consumer, or {@code null} to disable prefetching
   */
  void setPrefetchHook(IntConsumer hook) {
    this.prefetchHook = hook;
  }

  HnswSearcher(
      HnswGraph graph,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction,
      NodeScorerFactory scorerFactory) {
    this(graph, vectors, similarityFunction, scorerFactory, false);
  }

  /** Creates a searcher using full-precision scoring backed by this searcher's own scratch. */
  HnswSearcher(
      HnswGraph graph, RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
    this(graph, vectors, similarityFunction, null, true);
  }

  /**
   * Shared constructor. When {@code useDefaultScorer} is true, the searcher's {@code scorerFactory}
   * is bound to {@link #defaultScorer(float[])} (which closes over this searcher's reusable
   * scratch) and {@code factory} is ignored; otherwise the pluggable {@code factory} is used
   * verbatim.
   */
  private HnswSearcher(
      HnswGraph graph,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction,
      NodeScorerFactory factory,
      boolean useDefaultScorer) {
    this.graph = graph;
    this.vectors = vectors;
    this.similarityFunction = similarityFunction;
    int graphSize = Math.max(1, graph.size());
    this.visitedTag = new int[graphSize];
    this.visitedGen = 0;
    // Initial heap capacity of 256 matches the typical ef (50-200) plus headroom — a query that
    // hits the top of that range never needs to grow the heap. The NodeQueue grows geometrically
    // on overflow, so a wrong guess just costs one realloc per ef-doubling, not per-insert.
    this.candidates = new NodeQueue(256, false); // max-heap, beam-search frontier
    this.results = new NodeQueue(256, true); // min-heap, current top-k by score
    // Rescore is bounded by k (typically <= 100), so a small starting capacity is enough.
    this.rescoreHeap = new NodeQueue(64, true); // min-heap, grows as needed
    this.tmpNodes = new int[graphSize];
    this.tmpScores = new float[graphSize];
    // Fit the widest possible layer-0 neighbor list (2M+1 upper bound) but never below BULK_BATCH.
    this.bulkCapacity = Math.max(BULK_BATCH, graph.maxConnections0() + 2);
    this.bulkIds = new int[bulkCapacity];
    this.bulkScores = new float[bulkCapacity];

    // Scorer scratch is searcher-owned (reused across queries), sized to bulkCapacity so the fused
    // GEMV path can score a full neighbor-list batch in one call. Only the default scorer uses it.
    this.scorerPool = new float[bulkCapacity][];
    this.scorerOut = new float[bulkCapacity];
    this.usesDefaultScorer = useDefaultScorer;
    if (useDefaultScorer && vectors.supportsSegments()) {
      // Off-heap query segment allocated ONCE for this searcher's lifetime — refilled per query
      // (MemorySegment.copy) inside defaultScorer, never re-allocated per query.
      this.scorerArena = Arena.ofAuto();
      this.queryScratchSeg = scorerArena.allocate((long) vectors.dimension() * Float.BYTES);
      // Reusable row-segment scratch for the fused segment GEMV — sized to bulkCapacity like the
      // float[][] scorerPool, allocated ONCE here, refilled per bulkScore call.
      this.scorerRows = new MemorySegment[bulkCapacity];
    } else {
      this.scorerArena = null;
      this.queryScratchSeg = null;
      this.scorerRows = null;
    }
    this.scorerFactory = useDefaultScorer ? this::defaultScorer : factory;
  }

  /**
   * Default full-precision scorer bound to this searcher's reusable scratch (segment path: the
   * once-allocated {@link #queryScratchSeg}; fused path: {@link #scorerPool} / {@link #scorerOut}).
   * This replaces the former per-query allocation of an {@code Arena}/{@code float[64][]} pool.
   *
   * <p>When the underlying {@link RandomAccessVectors} returns stable references (i.e. {@code
   * !sharesReturnBuffer()} and no segments), the returned scorer overrides {@link
   * NodeScorer#bulkScore} with a fused GEMV path (via {@link FusedSimilarity}) that aliases
   * neighbor references into the reusable pool — amortising query loads across 4 rows at a time.
   *
   * <p>Not thread-safe: this scorer aliases searcher-owned scratch, so it must only be used by the
   * thread that owns this searcher. Multi-start workers build their own via their own searcher.
   */
  private NodeScorer defaultScorer(float[] query) {
    final SimilarityFunction sim = similarityFunction;
    final RandomAccessVectors v = vectors;
    if (v.supportsSegments()) {
      // Zero-copy path: SIMD-score directly from the mmap slice, no per-neighbor float[] copy. The
      // QUERY is uploaded into the searcher's reusable off-heap segment exactly ONCE per query
      // here (NOT inside score()), then every neighbor score reads the query segment plus a
      // zero-copy vectorSegment() view of the stored vector.
      final int dim = v.dimension();
      final MemorySegment qs = queryScratchSeg;
      MemorySegment.copy(query, 0, qs, ValueLayout.JAVA_FLOAT, 0L, dim);
      final MemorySegment[] rows = scorerRows;
      final float[] scratch = scorerOut;
      return new NodeScorer() {
        @Override
        public float score(int nodeId) {
          // Greedy/single path: score directly from the mmap slice, no float[] copy.
          return sim.compare(qs, v.vectorSegment(nodeId), dim);
        }

        @Override
        public void bulkScore(int[] nodeIds, int offset, int count, float[] outScores) {
          // Gather zero-copy row slices into the reusable per-searcher scratch (no per-call
          // allocation), then fused-GEMV score all of them with the query loaded once per 4 rows.
          for (int i = 0; i < count; i++) rows[i] = v.vectorSegment(nodeIds[offset + i]);
          FusedSimilarity.bulkCompareSegments(sim, query, rows, dim, scratch, outScores, count);
        }
      };
    }
    if (v.sharesReturnBuffer()) {
      // Shared-buffer impls can't safely be pooled for fused scoring; fall back to scalar.
      return nodeId -> sim.compare(query, v.getVector(nodeId));
    }
    final float[][] pool = scorerPool;
    final float[] out = scorerOut;
    return new NodeScorer() {
      @Override
      public float score(int nodeId) {
        return sim.compare(query, v.getVector(nodeId));
      }

      @Override
      public void bulkScore(int[] nodeIds, int offset, int count, float[] outScores) {
        for (int i = 0; i < count; i++) pool[i] = v.getVector(nodeIds[offset + i]);
        FusedSimilarity.bulkCompare(sim, query, pool, out, outScores, count);
      }
    };
  }

  /**
   * Searches the graph for the k nearest neighbors to the query vector.
   *
   * @param query the query vector
   * @param k number of results to return
   * @param efSearch beam width for layer 0 search (must be >= k)
   * @return the top-k results sorted by score descending
   */
  public SearchResult search(float[] query, int k, int efSearch) {
    if (efSearch < k) {
      throw new IllegalArgumentException("efSearch (" + efSearch + ") must be >= k (" + k + ")");
    }
    if (graph.size() == 0) {
      throw new IllegalStateException("Cannot search an empty graph");
    }

    NodeScorer scorer = scorerFactory.scorer(query);

    int ep = graph.entryNode();
    int maxLevel = graph.maxLevel();

    // Phase 1: Greedy descent from maxLevel to 1
    int currentBest = ep;
    for (int layer = maxLevel; layer >= 1; layer--) {
      currentBest = greedyDescend(currentBest, layer, scorer);
    }

    // Phase 2: Beam search at layer 0
    singleEntry[0] = currentBest;
    NeighborArray beamResults = beamSearch(singleEntry, efSearch, 0, scorer);

    return extractTopK(beamResults, k);
  }

  /** Searches with default efSearch = max(k, 100). */
  public SearchResult search(float[] query, int k) {
    return search(query, k, Math.max(k, 100));
  }

  /**
   * Multi-start parallel beam search at layer 0. Runs {@code nStarts} independent beam searches
   * from diverse seed nodes on virtual threads and merges their outputs into a single top-{@code k}
   * result.
   *
   * <p>Phase 1 greedy descent (maxLevel..1) runs single-threaded on this searcher. Seeds are
   * selected by a bounded layer-1 beam search with {@code ef = 2 * nStarts} whose top-{@code
   * nStarts} node ids become the parallel workers' entry points. When {@code nStarts <= 1} or the
   * graph has only layer 0 (no diverse seeds available), this delegates to {@link #search} and is
   * bit-identical to the single-start path.
   *
   * <p>Each worker allocates its own {@link HnswSearcher} so that {@code visited}, {@code
   * candidates}, and {@code results} scratch is per-thread. The merge step runs on the caller.
   *
   * <p><b>Determinism.</b> {@code nStarts == 1} is deterministic. {@code nStarts > 1} is NOT
   * deterministic across runs — the merge order depends on virtual-thread completion timing.
   *
   * @param query the query vector
   * @param k number of results to return
   * @param efSearch beam width per worker at layer 0 (must be >= k)
   * @param nStarts number of parallel seeds ({@code values <= 1} delegate to single-start search)
   * @return the top-k results sorted by score descending
   */
  public SearchResult searchMultiStart(float[] query, int k, int efSearch, int nStarts) {
    if (nStarts <= 1) {
      return search(query, k, efSearch);
    }
    if (efSearch < k) {
      throw new IllegalArgumentException("efSearch (" + efSearch + ") must be >= k (" + k + ")");
    }
    if (graph.size() == 0) {
      throw new IllegalStateException("Cannot search an empty graph");
    }

    NodeScorer scorer = scorerFactory.scorer(query);

    // Phase 1: greedy descent maxLevel..1 on this searcher's scratch.
    int currentBest = graph.entryNode();
    for (int layer = graph.maxLevel(); layer >= 1; layer--) {
      currentBest = greedyDescend(currentBest, layer, scorer);
    }

    // Phase 2: seed selection via bounded layer-1 beam search. If the graph has only layer 0
    // there are no long-range hop edges to diversify seeds — fall back to single-start.
    int[] seeds = pickSeeds(currentBest, scorer, nStarts);
    if (seeds.length <= 1) {
      return search(query, k, efSearch);
    }

    // Phase 3: spawn one worker per seed on a scoped virtual-thread executor.
    return runParallelBeams(query, k, efSearch, seeds);
  }

  /**
   * Selects up to {@code nStarts} diverse seed node ids by running a bounded layer-1 beam search
   * with {@code ef = 2 * nStarts} from {@code entry}. Returns the top-{@code min(nStarts, results)}
   * node ids in descending score order. Returns {@code {entry}} when the graph has only layer 0 (no
   * layer-1 neighbors exist to diversify from).
   */
  int[] pickSeeds(int entry, NodeScorer scorer, int nStarts) {
    if (graph.maxLevel() < 1) {
      return new int[] {entry};
    }
    int seedEf = Math.max(2, 2 * nStarts);
    singleEntry[0] = entry;
    NeighborArray seedResults = beamSearch(singleEntry, seedEf, 1, scorer);
    int count = Math.min(nStarts, seedResults.size());
    if (count <= 0) {
      return new int[] {entry};
    }
    int[] seeds = new int[count];
    for (int i = 0; i < count; i++) {
      seeds[i] = seedResults.node(i);
    }
    return seeds;
  }

  /**
   * Spawns one virtual-thread worker per seed, each running a fresh {@link HnswSearcher}'s layer-0
   * beam search, then merges their {@link NeighborArray} outputs into a single top-{@code k} {@link
   * SearchResult} using a size-bounded min-heap and a de-dup {@link BitSet}.
   */
  private SearchResult runParallelBeams(float[] query, int k, int efSearch, int[] seeds) {
    List<NeighborArray> perWorker = new ArrayList<>(seeds.length);
    try (var exec = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<NeighborArray>> futures = new ArrayList<>(seeds.length);
      for (int seed : seeds) {
        final int s = seed;
        futures.add(
            exec.submit(
                () -> {
                  // Each worker owns its scratch. When this searcher uses the default (stateful,
                  // scratch-bound) scorer, the worker MUST build its own default scorer via the
                  // 3-arg constructor — never share this searcher's query segment / pool across
                  // threads. Pluggable factories are stateless per query, so they are shared.
                  HnswSearcher worker =
                      usesDefaultScorer
                          ? new HnswSearcher(graph, vectors, similarityFunction)
                          : new HnswSearcher(graph, vectors, similarityFunction, scorerFactory);
                  NodeScorer ws = worker.scorerFactory.scorer(query);
                  worker.singleEntry[0] = s;
                  return worker.beamSearch(worker.singleEntry, efSearch, 0, ws);
                }));
      }
      for (Future<NeighborArray> f : futures) {
        try {
          perWorker.add(f.get());
        } catch (ExecutionException e) {
          // Cancel outstanding workers eagerly so the try-with-resources close() does not block
          // draining virtual threads that are no longer needed.
          exec.shutdownNow();
          throw new RuntimeException("searchMultiStart worker failed", e.getCause());
        } catch (InterruptedException e) {
          exec.shutdownNow();
          Thread.currentThread().interrupt();
          throw new RuntimeException("searchMultiStart interrupted", e);
        }
      }
    }

    // Merge: size-bounded min-heap + BitSet de-dup. NodeQueue requires non-negative scores, so we
    // use insertWithOverflow which already guards the bound; duplicates across workers are skipped.
    NodeQueue resultHeap = new NodeQueue(Math.max(1, k), true);
    BitSet sink = new BitSet(graph.size());
    for (NeighborArray na : perWorker) {
      for (int i = 0; i < na.size(); i++) {
        int id = na.node(i);
        if (sink.get(id)) continue;
        sink.set(id);
        resultHeap.insertWithOverflow(id, na.score(i), k);
      }
    }

    int resultSize = resultHeap.size();
    int[] nodeIds = new int[resultSize];
    float[] scores = new float[resultSize];
    for (int i = resultSize - 1; i >= 0; i--) {
      long entry = resultHeap.poll();
      nodeIds[i] = NodeQueue.nodeId(entry);
      scores[i] = NodeQueue.score(entry);
    }
    return new SearchResult(nodeIds, scores);
  }

  /**
   * Two-pass search: coarse quantized pass followed by full-precision rescore.
   *
   * <p>The coarse pass uses the scorer factory (typically quantized) to find candidates with a
   * larger beam width. The rescore pass re-evaluates candidates with full-precision vectors.
   *
   * @param query the query vector
   * @param k number of final results
   * @param efSearch beam width for coarse search
   * @param overQueryFactor multiplier for coarse-pass k (e.g., 2.0 means retrieve 2*k candidates)
   * @return the top-k results after full-precision rescoring
   */
  public SearchResult searchTwoPass(float[] query, int k, int efSearch, float overQueryFactor) {
    int coarseK = Math.max(k, (int) (k * overQueryFactor));
    int coarseEf = Math.max(coarseK, efSearch);
    SearchResult coarseResults = search(query, coarseK, coarseEf);
    return rescore(query, coarseResults.nodeIds(), k);
  }

  /** Two-pass search with default efSearch=max(k,100) and overQueryFactor=2.0. */
  public SearchResult searchTwoPass(float[] query, int k) {
    return searchTwoPass(query, k, Math.max(k, 100), 2.0f);
  }

  /**
   * ACORN-style pre-filtered search: navigates the full graph but accumulates only nodes that pass
   * {@code predicate} into the result set.
   *
   * <p>Non-matching nodes are still explored for <em>navigation</em> — they help reach matching
   * nodes that would otherwise be disconnected under the filter. This is the key insight from the
   * ACORN paper (Nori et al., 2023): graph edges are treated as routing infrastructure, not as
   * membership assertions.
   *
   * <p>The beam width {@code efSearch} governs the size of the matching result set, not the total
   * number of nodes visited. For highly selective filters, more non-matching nodes will be
   * traversed before {@code efSearch} matching results are gathered.
   *
   * @param query the query vector
   * @param k number of matching results to return
   * @param efSearch beam width (number of matching candidates to accumulate; must be ≥ k)
   * @param predicate ordinal-level filter — {@code true} means the node is eligible for results
   * @return the top-k matching results sorted by score descending
   */
  public SearchResult searchFiltered(float[] query, int k, int efSearch, IntPredicate predicate) {
    if (efSearch < k) {
      throw new IllegalArgumentException("efSearch (" + efSearch + ") must be >= k (" + k + ")");
    }
    if (graph.size() == 0) {
      throw new IllegalStateException("Cannot search an empty graph");
    }

    NodeScorer scorer = scorerFactory.scorer(query);

    // Phase 1: Greedy descent (no filter — any node can serve as routing).
    int ep = graph.entryNode();
    int currentBest = ep;
    for (int layer = graph.maxLevel(); layer >= 1; layer--) {
      currentBest = greedyDescend(currentBest, layer, scorer);
    }

    // Phase 2: Predicate-aware beam search at layer 0.
    singleEntry[0] = currentBest;
    NeighborArray beamResults = beamSearchFiltered(singleEntry, efSearch, 0, scorer, predicate);

    return extractTopK(beamResults, k);
  }

  /** Pre-filtered search with default efSearch = max(k, 100). */
  public SearchResult searchFiltered(float[] query, int k, IntPredicate predicate) {
    return searchFiltered(query, k, Math.max(k, 100), predicate);
  }

  /**
   * Rescores candidate node IDs with full-precision vectors and returns the top-k.
   *
   * <p>Reuses the pre-allocated {@code rescoreHeap} scratch buffer — no heap allocation per call.
   *
   * @param query the query vector
   * @param candidateNodeIds node IDs from the coarse pass
   * @param k number of results to return
   * @return the top-k results after full-precision rescoring, sorted by score descending
   */
  public SearchResult rescore(float[] query, int[] candidateNodeIds, int k) {
    rescoreHeap.clear();
    for (int nodeId : candidateNodeIds) {
      float score = similarityFunction.compare(query, vectors.getVector(nodeId));
      rescoreHeap.insertWithOverflow(nodeId, score, k);
    }

    int resultSize = rescoreHeap.size();
    int[] nodeIds = new int[resultSize];
    float[] scores = new float[resultSize];
    // Drain min-heap in reverse to produce descending score order.
    for (int i = resultSize - 1; i >= 0; i--) {
      long entry = rescoreHeap.poll();
      nodeIds[i] = NodeQueue.nodeId(entry);
      scores[i] = NodeQueue.score(entry);
    }
    return new SearchResult(nodeIds, scores);
  }

  /**
   * Greedy descent: walk to the best neighbor at the given layer.
   *
   * <p>Neighbor ids are gathered into the searcher's reusable {@code bulkIds} scratch and scored in
   * a single {@link NodeScorer#bulkScore} call, routing descent through the fused (in-memory) /
   * zero-copy-segment (mmap) path instead of one virtual {@code score()} per neighbor. Semantics
   * are identical to the per-node loop: we move to the highest-scoring neighbor that beats {@code
   * currentScore} (first occurrence on ties — the score comparison is strict {@code >}), and keep
   * walking while an improvement was found.
   *
   * <p>A layer-&ge;1 neighbor list is bounded by {@code graph.maxConnections() <= bulkCapacity},
   * but {@code count} is clamped to {@code bulkCapacity} defensively; any overflow tail falls back
   * to per-node {@code score()}.
   */
  int greedyDescend(int entryPoint, int layer, NodeScorer scorer) {
    int current = entryPoint;
    float currentScore = scorer.score(current);

    boolean improved = true;
    while (improved) {
      improved = false;
      NeighborArray neighbors = graph.getNeighbors(current, layer);
      if (neighbors == null) break;

      int n = neighbors.size();
      int batch = Math.min(n, bulkCapacity);
      for (int i = 0; i < batch; i++) {
        bulkIds[i] = neighbors.node(i);
      }
      scorer.bulkScore(bulkIds, 0, batch, bulkScores);

      int bestId = current;
      float bestScore = currentScore;
      for (int i = 0; i < batch; i++) {
        if (bulkScores[i] > bestScore) {
          bestScore = bulkScores[i];
          bestId = neighbors.node(i);
        }
      }
      // Defensive overflow tail: only reached if a neighbor list exceeds bulkCapacity.
      for (int i = batch; i < n; i++) {
        int neighborId = neighbors.node(i);
        float score = scorer.score(neighborId);
        if (score > bestScore) {
          bestScore = score;
          bestId = neighborId;
        }
      }

      if (bestId != current) {
        current = bestId;
        currentScore = bestScore;
        improved = true;
      }
    }
    return current;
  }

  /**
   * Begins a new visited generation in O(1) amortized time (replacing an O(graph.size()) clear). On
   * the rare {@code int} wraparound, resets the tag array so a stale tag can't alias the new
   * generation.
   */
  private void nextVisitedGeneration() {
    if (++visitedGen == 0) {
      java.util.Arrays.fill(visitedTag, 0);
      visitedGen = 1;
    }
  }

  /** Beam search at a given layer using candidate max-heap and result min-heap. */
  NeighborArray beamSearch(int[] entryPoints, int ef, int layer, NodeScorer scorer) {
    nextVisitedGeneration();
    candidates.clear();
    results.clear();

    // Seed with entry points
    for (int ep : entryPoints) {
      if (visitedTag[ep] != visitedGen) {
        visitedTag[ep] = visitedGen;
        float score = scorer.score(ep);
        candidates.add(ep, score);
        results.add(ep, score);
      }
    }

    // Hoist the neighbor-batch capability check once: it's a virtual call on a per-search-fresh
    // scorer and would otherwise sit inside the hot while loop. Only Fused ADC scorers return
    // true here, and only at layer 0 (packed codes are a leaf-layer optimization).
    final boolean useNeighborBatch = layer == 0 && scorer.supportsNeighborBatch();

    // Beam search
    while (!candidates.isEmpty()) {
      long topCandidate = candidates.poll();
      float candidateScore = NodeQueue.score(topCandidate);
      int candidateId = NodeQueue.nodeId(topCandidate);

      // Early termination
      if (results.size() >= ef) {
        float worstResult = NodeQueue.score(results.peek());
        if (candidateScore < worstResult) {
          break;
        }
      }

      NeighborArray neighbors = graph.getNeighbors(candidateId, layer);
      if (neighbors == null) continue;
      nodesVisited++;

      // SSD prefetch pass: issue async touch-reads for all neighbors before scoring.
      // For mmap-backed vector stores this causes the OS to page-in the relevant 4 KiB pages
      // concurrently with any remaining scoring work on the current candidate's neighbors.
      if (prefetchHook != null) {
        for (int i = 0; i < neighbors.size(); i++) {
          prefetchHook.accept(neighbors.node(i));
        }
      }

      int n = neighbors.size();
      if (useNeighborBatch) {
        // Fused ADC path: score the whole neighbor list with one stride-1 sweep over the
        // origin's packed code layout. Visited neighbors are filtered post-score — negligible
        // overhead since the ADC cost scales with M (≤ 32) rather than dim.
        scorer.scoreNeighborBatch(candidateId, n, bulkScores);
        for (int i = 0; i < n; i++) {
          int neighborId = neighbors.node(i);
          if (visitedTag[neighborId] == visitedGen) continue;
          visitedTag[neighborId] = visitedGen;
          ingestOne(neighborId, bulkScores[i], ef);
        }
      } else {
        // Gather unvisited neighbors into a batch, then score them all in one fused call.
        int batchCount = 0;
        for (int i = 0; i < n; i++) {
          int neighborId = neighbors.node(i);
          if (visitedTag[neighborId] == visitedGen) continue;
          visitedTag[neighborId] = visitedGen;
          bulkIds[batchCount++] = neighborId;
          if (batchCount == BULK_BATCH) {
            scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
            ingestBatch(batchCount, ef);
            batchCount = 0;
          }
        }
        if (batchCount > 0) {
          scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
          ingestBatch(batchCount, ef);
        }
      }
    }

    // Convert results min-heap to sorted NeighborArray (descending).
    // Reuse pre-allocated tmpNodes/tmpScores scratch fields — no allocation per search.
    int resultSize = results.size();
    NeighborArray resultArray = obtainResultArray(resultSize == 0 ? 1 : resultSize);
    for (int i = resultSize - 1; i >= 0; i--) {
      long entry = results.poll();
      tmpNodes[i] = NodeQueue.nodeId(entry);
      tmpScores[i] = NodeQueue.score(entry);
    }
    for (int i = 0; i < resultSize; i++) {
      resultArray.insert(tmpNodes[i], tmpScores[i]);
    }

    return resultArray;
  }

  /**
   * Predicate-aware beam search at a given layer (ACORN navigation pattern).
   *
   * <p>All unvisited neighbours are added to the exploration {@code candidates} heap regardless of
   * the predicate — this is the "routing through non-matching nodes" property that distinguishes
   * ACORN from simple post-filtering. Only nodes that pass {@code predicate} are admitted to the
   * {@code results} min-heap.
   *
   * <p>Early termination: once the {@code results} heap contains {@code ef} matching entries, we
   * stop as soon as the best remaining candidate cannot beat the worst matching result — identical
   * to the unfiltered {@link #beamSearch} condition. This may under-explore for very selective
   * filters, but provides the same asymptotic guarantee as the standard algorithm.
   */
  NeighborArray beamSearchFiltered(
      int[] entryPoints, int ef, int layer, NodeScorer scorer, IntPredicate predicate) {
    nextVisitedGeneration();
    candidates.clear();
    results.clear();

    // Seed with entry points.
    for (int ep : entryPoints) {
      if (visitedTag[ep] != visitedGen) {
        visitedTag[ep] = visitedGen;
        float score = scorer.score(ep);
        candidates.add(ep, score);
        // Only matching entry points go into results.
        if (predicate.test(ep)) {
          results.add(ep, score);
        }
      }
    }

    // Hoist the neighbor-batch capability check once (see beamSearch for rationale).
    final boolean useNeighborBatch = layer == 0 && scorer.supportsNeighborBatch();

    while (!candidates.isEmpty()) {
      long topCandidate = candidates.poll();
      float candidateScore = NodeQueue.score(topCandidate);
      int candidateId = NodeQueue.nodeId(topCandidate);

      // Early termination — same condition as unfiltered beamSearch.
      if (results.size() >= ef) {
        float worstResult = NodeQueue.score(results.peek());
        if (candidateScore < worstResult) {
          break;
        }
      }

      NeighborArray neighbors = graph.getNeighbors(candidateId, layer);
      if (neighbors == null) continue;

      // SSD prefetch pass: issue async touch-reads for all neighbors before scoring.
      if (prefetchHook != null) {
        for (int i = 0; i < neighbors.size(); i++) {
          prefetchHook.accept(neighbors.node(i));
        }
      }

      int n = neighbors.size();
      if (useNeighborBatch) {
        // Fused ADC path: score every neighbor in one call, then filter visited and apply
        // the predicate. Non-matching but unvisited neighbors still route through {@code
        // candidates} to preserve ACORN navigation semantics.
        scorer.scoreNeighborBatch(candidateId, n, bulkScores);
        for (int i = 0; i < n; i++) {
          int neighborId = neighbors.node(i);
          if (visitedTag[neighborId] == visitedGen) continue;
          visitedTag[neighborId] = visitedGen;
          ingestOneFiltered(neighborId, bulkScores[i], ef, predicate);
        }
      } else {
        // Gather unvisited neighbors into a batch, then score them all in one fused call.
        int batchCount = 0;
        for (int i = 0; i < n; i++) {
          int neighborId = neighbors.node(i);
          if (visitedTag[neighborId] == visitedGen) continue;
          visitedTag[neighborId] = visitedGen;
          bulkIds[batchCount++] = neighborId;
          if (batchCount == BULK_BATCH) {
            scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
            ingestBatchFiltered(batchCount, ef, predicate);
            batchCount = 0;
          }
        }
        if (batchCount > 0) {
          scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
          ingestBatchFiltered(batchCount, ef, predicate);
        }
      }
    }

    // Convert results min-heap to sorted NeighborArray (descending).
    int resultSize = results.size();
    NeighborArray resultArray = obtainResultArray(resultSize == 0 ? 1 : resultSize);
    for (int i = resultSize - 1; i >= 0; i--) {
      long entry = results.poll();
      tmpNodes[i] = NodeQueue.nodeId(entry);
      tmpScores[i] = NodeQueue.score(entry);
    }
    for (int i = 0; i < resultSize; i++) {
      resultArray.insert(tmpNodes[i], tmpScores[i]);
    }

    return resultArray;
  }

  /** Ingests a single scored neighbor into the unfiltered beam-search heaps. */
  private void ingestOne(int neighborId, float score, int ef) {
    // Single sift-down eviction; only explore the neighbor if it entered the result beam.
    if (results.insertWithOverflow(neighborId, score, ef)) {
      candidates.add(neighborId, score);
    }
  }

  /**
   * Ingests a fused-scored batch into the unfiltered beam-search heaps. Reads ids from {@code
   * bulkIds[0..count)} and their scores from {@code bulkScores[0..count)}.
   */
  private void ingestBatch(int count, int ef) {
    for (int i = 0; i < count; i++) {
      ingestOne(bulkIds[i], bulkScores[i], ef);
    }
  }

  /**
   * Ingests a single scored neighbor into the predicate-aware beam-search heaps (ACORN pattern).
   * All neighbors route through {@code candidates}; only predicate-passing ones enter {@code
   * results}.
   */
  private void ingestOneFiltered(int neighborId, float score, int ef, IntPredicate predicate) {
    candidates.add(neighborId, score);
    if (predicate.test(neighborId)) {
      // Single sift-down eviction on the bounded result heap.
      results.insertWithOverflow(neighborId, score, ef);
    }
  }

  /**
   * Ingests a fused-scored batch into the predicate-aware beam-search heaps (ACORN pattern). All
   * neighbours route through {@code candidates}; only predicate-passing ones enter {@code results}.
   */
  private void ingestBatchFiltered(int count, int ef, IntPredicate predicate) {
    for (int i = 0; i < count; i++) {
      ingestOneFiltered(bulkIds[i], bulkScores[i], ef, predicate);
    }
  }

  /**
   * Returns the reusable beam-search result array cleared and sized to hold at least {@code
   * capacity} entries. Grown (reallocated) only when the previous array is too small; otherwise the
   * same array is cleared and reused. See the {@link #reusableResult} field for the consumption
   * safety invariant.
   */
  private NeighborArray obtainResultArray(int capacity) {
    if (reusableResult == null || reusableResult.maxSize() < capacity) {
      reusableResult = new NeighborArray(capacity);
    } else {
      reusableResult.clear();
    }
    return reusableResult;
  }

  /** Extracts the top-k results from a sorted NeighborArray. */
  private SearchResult extractTopK(NeighborArray beamResults, int k) {
    int resultCount = Math.min(k, beamResults.size());
    int[] nodeIds = new int[resultCount];
    float[] scores = new float[resultCount];
    for (int i = 0; i < resultCount; i++) {
      nodeIds[i] = beamResults.node(i);
      scores[i] = beamResults.score(i);
    }
    return new SearchResult(nodeIds, scores);
  }
}
