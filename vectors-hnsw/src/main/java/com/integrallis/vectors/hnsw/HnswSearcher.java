package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.BitSet;

/**
 * Hierarchical search over an HNSW graph. Performs greedy descent through upper layers followed by
 * beam search at layer 0.
 *
 * <p>Not thread-safe — owns scratch buffers (NodeQueue, BitSet). For concurrent queries, create a
 * separate {@code HnswSearcher} per thread via {@link HnswIndex#searcher()}.
 */
public final class HnswSearcher {

  private final HnswGraph graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;

  // Scratch buffers reused across searches
  private final BitSet visited;
  private final NodeQueue candidates;
  private final NodeQueue results;

  HnswSearcher(
      HnswGraph graph, RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
    this.graph = graph;
    this.vectors = vectors;
    this.similarityFunction = similarityFunction;
    this.visited = new BitSet(graph.size());
    this.candidates = new NodeQueue(256, false); // max-heap
    this.results = new NodeQueue(256, true); // min-heap
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

    int ep = graph.entryNode();
    int maxLevel = graph.maxLevel();

    // Phase 1: Greedy descent from maxLevel to 1
    int currentBest = ep;
    for (int layer = maxLevel; layer >= 1; layer--) {
      currentBest = greedyDescend(query, currentBest, layer);
    }

    // Phase 2: Beam search at layer 0
    NeighborArray beamResults = beamSearch(query, new int[] {currentBest}, efSearch, 0);

    // Extract top-k
    int resultCount = Math.min(k, beamResults.size());
    int[] nodeIds = new int[resultCount];
    float[] scores = new float[resultCount];
    for (int i = 0; i < resultCount; i++) {
      nodeIds[i] = beamResults.node(i);
      scores[i] = beamResults.score(i);
    }
    return new SearchResult(nodeIds, scores);
  }

  /** Searches with default efSearch = max(k, 100). */
  public SearchResult search(float[] query, int k) {
    return search(query, k, Math.max(k, 100));
  }

  /** Greedy descent: walk to the best neighbor at the given layer. */
  int greedyDescend(float[] query, int entryPoint, int layer) {
    int current = entryPoint;
    float currentScore = similarityFunction.compare(query, vectors.getVector(current));

    boolean improved = true;
    while (improved) {
      improved = false;
      NeighborArray neighbors = graph.getNeighbors(current, layer);
      if (neighbors == null) break;

      for (int i = 0; i < neighbors.size(); i++) {
        int neighborId = neighbors.node(i);
        float score = similarityFunction.compare(query, vectors.getVector(neighborId));
        if (score > currentScore) {
          current = neighborId;
          currentScore = score;
          improved = true;
        }
      }
    }
    return current;
  }

  /** Beam search at a given layer using candidate max-heap and result min-heap. */
  NeighborArray beamSearch(float[] query, int[] entryPoints, int ef, int layer) {
    visited.clear();
    candidates.clear();
    results.clear();

    // Seed with entry points
    for (int ep : entryPoints) {
      if (!visited.get(ep)) {
        visited.set(ep);
        float score = similarityFunction.compare(query, vectors.getVector(ep));
        candidates.add(ep, score);
        results.add(ep, score);
      }
    }

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

      for (int i = 0; i < neighbors.size(); i++) {
        int neighborId = neighbors.node(i);
        if (visited.get(neighborId)) continue;
        visited.set(neighborId);

        float score = similarityFunction.compare(query, vectors.getVector(neighborId));

        if (results.size() < ef) {
          candidates.add(neighborId, score);
          results.add(neighborId, score);
        } else {
          float worstResult = NodeQueue.score(results.peek());
          if (score > worstResult) {
            candidates.add(neighborId, score);
            results.poll();
            results.add(neighborId, score);
          }
        }
      }
    }

    // Convert results min-heap to sorted NeighborArray (descending)
    int resultSize = results.size();
    var resultArray = new NeighborArray(resultSize == 0 ? 1 : resultSize);
    int[] tmpNodes = new int[resultSize];
    float[] tmpScores = new float[resultSize];
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
}
