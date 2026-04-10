package com.integrallis.vectors.vamana;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.BitSet;
import java.util.Objects;

/**
 * Greedy beam search on a flat Vamana graph. Unlike HNSW, there is no layer descent — search starts
 * at the medoid and expands through the single-layer graph.
 *
 * <p>Instances are <b>not thread-safe</b>. Use one searcher per thread (e.g., via {@link
 * ThreadLocal} in {@link VamanaIndex}).
 */
public final class VamanaSearcher {

  private final VamanaGraph graph;
  private final NodeScorerFactory scorerFactory;
  // -1 when a custom NodeScorerFactory is supplied (dimension unknown to this class).
  private final int dimension;

  // Pre-allocated scratch buffers
  private final NodeQueue candidates; // max-heap: best candidate on top for exploration
  private final NodeQueue results; // min-heap: worst result on top for eviction
  private BitSet visited;

  /**
   * Creates a searcher with full-precision scoring.
   *
   * @param graph the Vamana graph to search
   * @param vectors vector data for distance computation
   * @param sim similarity function
   */
  VamanaSearcher(VamanaGraph graph, RandomAccessVectors vectors, SimilarityFunction sim) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(sim, "sim must not be null");
    this.scorerFactory = query -> nodeId -> sim.compare(query, vectors.getVector(nodeId));
    this.dimension = vectors.dimension();
    int initialCapacity = Math.max(64, graph.size());
    this.candidates = new NodeQueue(initialCapacity, false); // max-heap
    this.results = new NodeQueue(initialCapacity, true); // min-heap
    this.visited = new BitSet(Math.max(1, graph.size()));
  }

  /**
   * Creates a searcher with a custom scorer factory (e.g., quantized scoring).
   *
   * @param graph the Vamana graph to search
   * @param scorerFactory factory for creating per-query scorers
   */
  VamanaSearcher(VamanaGraph graph, NodeScorerFactory scorerFactory) {
    this.graph = Objects.requireNonNull(graph, "graph must not be null");
    this.scorerFactory = Objects.requireNonNull(scorerFactory, "scorerFactory must not be null");
    this.dimension = -1; // unknown when using a custom scorer factory
    int initialCapacity = Math.max(64, graph.size());
    this.candidates = new NodeQueue(initialCapacity, false); // max-heap
    this.results = new NodeQueue(initialCapacity, true); // min-heap
    this.visited = new BitSet(Math.max(1, graph.size()));
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
    if (dimension >= 0 && query.length != dimension) {
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
}
