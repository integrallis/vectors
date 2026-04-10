package com.integrallis.vectors.vamana;

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

  private final VamanaGraph graph;
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
      VamanaGraph graph,
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
  }

  /**
   * Creates a searcher with full-precision scoring. Convenience constructor that builds the default
   * {@link NodeScorerFactory} from {@code vectors} and {@code sim}.
   *
   * @param graph the Vamana graph to search
   * @param vectors vector data for distance computation
   * @param sim similarity function
   */
  VamanaSearcher(VamanaGraph graph, RandomAccessVectors vectors, SimilarityFunction sim) {
    this(graph, vectors, sim, query -> nodeId -> sim.compare(query, vectors.getVector(nodeId)));
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

      // Expand neighbors
      NeighborArray neighbors = graph.getNeighbors(candidateId);
      for (int i = 0; i < neighbors.size(); i++) {
        int neighborId = neighbors.node(i);
        if (visited.get(neighborId)) {
          continue;
        }
        visited.set(neighborId);

        float neighborScore = scorer.score(neighborId);

        // Add to results if room or better than worst
        if (results.size() < L) {
          results.add(neighborId, neighborScore);
          candidates.add(neighborId, neighborScore);
        } else if (neighborScore > NodeQueue.score(results.peek())) {
          results.poll();
          results.add(neighborId, neighborScore);
          candidates.add(neighborId, neighborScore);
        }
      }
    }

    // Extract top-k from results (min-heap → drain and reverse)
    int resultSize = Math.min(k, results.size());
    int[] nodeIds = new int[resultSize];
    float[] scores = new float[resultSize];
    // Drain more than k if needed (results may have L entries)
    while (results.size() > resultSize) {
      results.poll();
    }
    // Now drain the remaining resultSize entries (worst first from min-heap)
    for (int i = resultSize - 1; i >= 0; i--) {
      long entry = results.poll();
      nodeIds[i] = NodeQueue.nodeId(entry);
      scores[i] = NodeQueue.score(entry);
    }

    return new SearchResult(nodeIds, scores);
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
