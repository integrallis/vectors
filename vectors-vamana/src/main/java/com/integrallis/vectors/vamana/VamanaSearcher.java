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
package com.integrallis.vectors.vamana;

import com.integrallis.vectors.core.FusedSimilarity;
import com.integrallis.vectors.core.SimilarityFunction;
import java.util.BitSet;
import java.util.Objects;

/**
 * Greedy beam search on a flat Vamana graph. Unlike HNSW, there is no layer descent — search starts
 * at the medoid and expands through the single-layer graph.
 *
 * <p>Scoring is delegated to a {@link NodeScorerFactory}, enabling both full-precision and
 * quantized scoring to share the same search algorithm.
 *
 * <p>Instances are <b>not thread-safe</b>. Use one searcher per thread (e.g., via {@link
 * ThreadLocal} in {@link VamanaIndex}).
 */
public final class VamanaSearcher {

  private final VamanaTopology graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;
  private final NodeScorerFactory scorerFactory;
  private final int dimension;

  // Pre-allocated scratch buffers
  private final NodeQueue candidates; // max-heap: best candidate on top for exploration
  private final NodeQueue results; // min-heap: worst result on top for eviction
  // Scratch buffer for rescore(): avoids allocating a new NodeQueue on every two-pass call.
  private final NodeQueue rescoreHeap;
  private BitSet visited;
  // Reusable buffer holding one node's neighbour ids, sized to the graph degree R. Lets the
  // topology fill ids without allocating (and lets a paged topology read straight from the mmap).
  private final int[] neighborScratch;
  // Scratch buffers for fused bulk neighbor scoring. Sized to an upper bound on graph degree R.
  //
  // Why 128 (vs HNSW's 64): Vamana's typical degree R is 64-128 (default DEFAULT_VAMANA_R=64,
  // tuned configs go to 128); HNSW's typical layer-0 degree is 2*M = 16-32 (M default 8-16).
  // The fused-scan working set is (R neighbors) × (dim * 4 bytes) for the raw-FP32 path. At
  // R=128, dim=768 that's 384 KB — fits L2 on every contemporary part but pushes past L1.
  // The Vamana scan path is graph-only (no PQ-table prefetch competing for L1), so the larger
  // batch amortises dispatch overhead better than HNSW's narrower window without hurting cache.
  private static final int BULK_BATCH = 128;
  private final int[] bulkIds = new int[BULK_BATCH];
  private final float[] bulkScores = new float[BULK_BATCH];

  /**
   * Creates a searcher with a custom {@link NodeScorerFactory}.
   *
   * <p>The {@code vectors} and {@code similarityFunction} are retained for full-precision rescoring
   * in {@link #rescore(float[], int[], int)} — the search algorithm itself uses only the scorer
   * factory.
   *
   * @param graph the Vamana graph to search
   * @param vectors vector data for full-precision rescoring
   * @param similarityFunction similarity function used for rescoring
   * @param scorerFactory factory that produces per-query scorers (e.g., full-precision or
   *     quantized)
   */
  VamanaSearcher(
      VamanaTopology graph,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction,
      NodeScorerFactory scorerFactory) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
    this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
    this.similarityFunction =
        Objects.requireNonNull(similarityFunction, "similarityFunction must not be null");
    this.scorerFactory = Objects.requireNonNull(scorerFactory, "scorerFactory must not be null");
    this.dimension = vectors.dimension();
    int initialCapacity = Math.max(64, graph.size());
    this.candidates = new NodeQueue(initialCapacity, false); // max-heap
    this.results = new NodeQueue(initialCapacity, true); // min-heap
    this.rescoreHeap = new NodeQueue(64, true); // min-heap, grows as needed
    this.visited = new BitSet(Math.max(1, graph.size()));
    this.neighborScratch = new int[Math.max(1, graph.maxDegree())];
  }

  /**
   * Creates a searcher with full-precision scoring. Convenience constructor that builds the default
   * {@link NodeScorerFactory} from {@code vectors} and {@code sim}.
   *
   * @param graph the Vamana graph to search
   * @param vectors vector data for distance computation
   * @param sim similarity function
   */
  VamanaSearcher(VamanaTopology graph, RandomAccessVectors vectors, SimilarityFunction sim) {
    this(graph, vectors, sim, defaultFullPrecisionFactory(vectors, sim));
  }

  /**
   * Default full-precision scorer factory. When the underlying {@link RandomAccessVectors} returns
   * stable references (i.e. {@code !sharesReturnBuffer()}), the returned scorer overrides {@link
   * NodeScorer#bulkScore} with a fused GEMV path (via {@link FusedSimilarity}) that aliases
   * neighbor references into a reusable pool — amortising query loads across 4 rows at a time.
   */
  private static NodeScorerFactory defaultFullPrecisionFactory(
      RandomAccessVectors vectors, SimilarityFunction sim) {
    if (vectors.supportsSegments()) {
      // Zero-copy path (mmap-backed, e.g. MemorySegmentRandomAccessVectors): score directly off the
      // stored vector's MemorySegment slice — no per-candidate mmap→float[] copy. The scratch is
      // allocated ONCE here (this factory is built once per searcher) and reused across queries; the
      // query is uploaded into an off-heap segment exactly once per query, not per candidate. Ports
      // the segment scorer already used by HnswSearcher.
      final int dim = vectors.dimension();
      final java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofAuto();
      final java.lang.foreign.MemorySegment queryScratchSeg =
          arena.allocate((long) dim * Float.BYTES);
      final java.lang.foreign.MemorySegment[] rows = new java.lang.foreign.MemorySegment[BULK_BATCH];
      final float[] segScratch = new float[BULK_BATCH];
      return query -> {
        java.lang.foreign.MemorySegment.copy(
            query, 0, queryScratchSeg, java.lang.foreign.ValueLayout.JAVA_FLOAT, 0L, dim);
        return new NodeScorer() {
          @Override
          public float score(int nodeId) {
            return sim.compare(queryScratchSeg, vectors.vectorSegment(nodeId), dim);
          }

          @Override
          public void bulkScore(int[] nodeIds, int offset, int count, float[] outScores) {
            for (int i = 0; i < count; i++) rows[i] = vectors.vectorSegment(nodeIds[offset + i]);
            FusedSimilarity.bulkCompareSegments(sim, query, rows, dim, segScratch, outScores, count);
          }
        };
      };
    }
    if (vectors.sharesReturnBuffer()) {
      return query -> nodeId -> sim.compare(query, vectors.getVector(nodeId));
    }
    return query ->
        new NodeScorer() {
          private final float[][] pool = new float[BULK_BATCH][];
          private final float[] out = new float[BULK_BATCH];

          @Override
          public float score(int nodeId) {
            return sim.compare(query, vectors.getVector(nodeId));
          }

          @Override
          public void bulkScore(int[] nodeIds, int offset, int count, float[] outScores) {
            for (int i = 0; i < count; i++) pool[i] = vectors.getVector(nodeIds[offset + i]);
            FusedSimilarity.bulkCompare(sim, query, pool, out, outScores, count);
          }
        };
  }

  /**
   * Searches for the k nearest neighbors with default search list size.
   *
   * @param query the query vector
   * @param k number of results to return
   * @return search result with ranked node IDs and scores
   */
  public SearchResult search(float[] query, int k) {
    return search(query, k, Math.max(k, 100));
  }

  /**
   * Searches for the k nearest neighbors.
   *
   * @param query the query vector
   * @param k number of results to return
   * @param searchListSize beam width L (larger = more accurate, slower)
   * @return search result with ranked node IDs and scores
   */
  public SearchResult search(float[] query, int k, int searchListSize) {
    Objects.requireNonNull(query, "query must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    int L = Math.max(searchListSize, k);

    // Reset scratch buffers.
    // Use visited.size() (allocated capacity, stable after clear()) NOT visited.length()
    // (logical size = index of highest set bit + 1, which returns 0 after clear() and
    // would trigger a new BitSet allocation on every search call after the first).
    candidates.clear();
    results.clear();
    if (visited.size() < graph.size()) {
      visited = new BitSet(graph.size());
    } else {
      visited.clear();
    }

    NodeScorer scorer = scorerFactory.scorer(query);

    // Start from medoid
    int entryPoint = graph.medoid();
    float entryScore = scorer.score(entryPoint);
    candidates.add(entryPoint, entryScore);
    results.add(entryPoint, entryScore);
    visited.set(entryPoint);

    // Beam search
    while (!candidates.isEmpty()) {
      long topCandidate = candidates.poll();
      int candidateId = NodeQueue.nodeId(topCandidate);
      float candidateScore = NodeQueue.score(topCandidate);

      // If the best candidate is worse than the worst result, we're done
      if (results.size() >= L && candidateScore < NodeQueue.score(results.peek())) {
        break;
      }

      // Expand neighbors — gather unvisited into a batch, then fused-score in one call. The
      // topology fills neighborScratch in stored order (heap copy or a paged read from the mmap).
      int n = graph.neighbors(candidateId, neighborScratch);
      int batchCount = 0;
      for (int i = 0; i < n; i++) {
        int neighborId = neighborScratch[i];
        if (visited.get(neighborId)) continue;
        visited.set(neighborId);
        bulkIds[batchCount++] = neighborId;
        if (batchCount == BULK_BATCH) {
          scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
          ingestBatch(batchCount, L);
          batchCount = 0;
        }
      }
      if (batchCount > 0) {
        scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
        ingestBatch(batchCount, L);
      }
    }

    return drainTopK(k);
  }

  /**
   * Pre-filtered beam search (ACORN-style): traverses through <i>all</i> neighbours to preserve
   * graph connectivity/routing, but only admits ordinals accepted by {@code predicate} into the
   * result set. For a high-selectivity filter this keeps recall high where an over-fetch
   * post-filter would collapse it (I.5). Mirrors {@code HnswSearcher.searchFiltered}.
   *
   * @param predicate accepts an ordinal iff it is eligible for the result set (e.g. matches the
   *     metadata filter and is not tombstoned)
   */
  public SearchResult searchFiltered(
      float[] query, int k, int searchListSize, java.util.function.IntPredicate predicate) {
    Objects.requireNonNull(query, "query must not be null");
    Objects.requireNonNull(predicate, "predicate must not be null");
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive: " + k);
    }
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension " + query.length + " does not match index dimension " + dimension);
    }
    int L = Math.max(searchListSize, k);

    candidates.clear();
    results.clear();
    if (visited.size() < graph.size()) {
      visited = new BitSet(graph.size());
    } else {
      visited.clear();
    }

    NodeScorer scorer = scorerFactory.scorer(query);

    int entryPoint = graph.medoid();
    if (entryPoint < 0) {
      return new SearchResult(new int[0], new float[0]); // empty graph
    }
    float entryScore = scorer.score(entryPoint);
    visited.set(entryPoint);
    // The medoid always routes; it only enters results if it matches the predicate.
    candidates.add(entryPoint, entryScore);
    if (predicate.test(entryPoint)) {
      results.add(entryPoint, entryScore);
    }

    while (!candidates.isEmpty()) {
      long topCandidate = candidates.poll();
      int candidateId = NodeQueue.nodeId(topCandidate);
      float candidateScore = NodeQueue.score(topCandidate);

      if (results.size() >= L && candidateScore < NodeQueue.score(results.peek())) {
        break;
      }

      int n = graph.neighbors(candidateId, neighborScratch);
      int batchCount = 0;
      for (int i = 0; i < n; i++) {
        int neighborId = neighborScratch[i];
        if (visited.get(neighborId)) continue;
        visited.set(neighborId);
        bulkIds[batchCount++] = neighborId;
        if (batchCount == BULK_BATCH) {
          scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
          ingestBatchFiltered(batchCount, L, predicate);
          batchCount = 0;
        }
      }
      if (batchCount > 0) {
        scorer.bulkScore(bulkIds, 0, batchCount, bulkScores);
        ingestBatchFiltered(batchCount, L, predicate);
      }
    }

    return drainTopK(k);
  }

  /** Extracts the top-{@code k} from the {@code results} min-heap (drain and reverse). */
  private SearchResult drainTopK(int k) {
    int resultSize = Math.min(k, results.size());
    int[] nodeIds = new int[resultSize];
    float[] scores = new float[resultSize];
    // Drain extra entries beyond k (results may hold up to L).
    while (results.size() > resultSize) {
      results.poll();
    }
    // Drain the remaining entries (worst first from the min-heap) into descending order.
    for (int i = resultSize - 1; i >= 0; i--) {
      long entry = results.poll();
      nodeIds[i] = NodeQueue.nodeId(entry);
      scores[i] = NodeQueue.score(entry);
    }
    return new SearchResult(nodeIds, scores);
  }

  /**
   * Ingests a fused-scored batch into the beam-search heaps. Reads ids from {@code
   * bulkIds[0..count)} and their scores from {@code bulkScores[0..count)}.
   */
  private void ingestBatch(int count, int L) {
    for (int i = 0; i < count; i++) {
      int neighborId = bulkIds[i];
      float neighborScore = bulkScores[i];
      // Single sift-down eviction; only explore the neighbor if it entered the result beam.
      if (results.insertWithOverflow(neighborId, neighborScore, L)) {
        candidates.add(neighborId, neighborScore);
      }
    }
  }

  /**
   * ACORN ingest: every scored neighbour is pushed to {@code candidates} (so traversal routes
   * through non-matching nodes), but only predicate-accepted ordinals enter {@code results}.
   */
  private void ingestBatchFiltered(int count, int L, java.util.function.IntPredicate predicate) {
    for (int i = 0; i < count; i++) {
      int neighborId = bulkIds[i];
      float neighborScore = bulkScores[i];
      candidates.add(neighborId, neighborScore); // always explore for routing
      if (!predicate.test(neighborId)) {
        continue;
      }
      if (results.size() < L) {
        results.add(neighborId, neighborScore);
      } else if (neighborScore > NodeQueue.score(results.peek())) {
        results.poll();
        results.add(neighborId, neighborScore);
      }
    }
  }

  /**
   * Two-pass search: coarse pass followed by full-precision rescore.
   *
   * <p>The coarse pass uses the scorer factory (typically quantized) to find candidates with a
   * larger beam width. The rescore pass re-evaluates candidates with full-precision vectors. When
   * the scorer factory is itself full-precision, both passes use exact scoring — the only effect is
   * over-fetching {@code overQueryFactor × k} candidates before trimming to {@code k}.
   *
   * @param query the query vector
   * @param k number of final results
   * @param searchListSize beam width L for the coarse pass
   * @param overQueryFactor multiplier for coarse-pass k (e.g., 2.0 retrieves 2*k candidates)
   * @return the top-k results after full-precision rescoring
   */
  public SearchResult searchTwoPass(
      float[] query, int k, int searchListSize, float overQueryFactor) {
    int coarseK = Math.max(k, (int) (k * overQueryFactor));
    int coarseL = Math.max(coarseK, searchListSize);
    SearchResult coarse = search(query, coarseK, coarseL);
    return rescore(query, coarse.nodeIds(), k);
  }

  /** Two-pass search with default searchListSize=max(k,100) and overQueryFactor=2.0. */
  public SearchResult searchTwoPass(float[] query, int k) {
    return searchTwoPass(query, k, Math.max(k, 100), 2.0f);
  }

  /**
   * Rescores candidate node IDs with full-precision vectors and returns the top-k.
   *
   * <p>Reuses the pre-allocated {@code rescoreHeap} scratch buffer — no heap allocation per call.
   *
   * <p><b>Not thread-safe.</b> The {@code rescoreHeap} field is shared mutable state; concurrent
   * callers on the same instance will corrupt each other's results. Each thread must own its own
   * {@link VamanaSearcher} (e.g., via the {@link ThreadLocal} pool managed by {@link VamanaIndex}).
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
}
