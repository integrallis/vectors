package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.BitSet;

/**
 * Hierarchical search over an HNSW graph. Performs greedy descent through upper layers followed by
 * beam search at layer 0.
 *
 * <p>Scoring is delegated to a {@link NodeScorerFactory}, enabling both full-precision and
 * quantized scoring to share the same search algorithm.
 *
 * <p>Not thread-safe — owns scratch buffers (NodeQueue, BitSet). For concurrent queries, create a
 * separate {@code HnswSearcher} per thread via {@link HnswIndex#searcher()}.
 */
public final class HnswSearcher {

  private final HnswGraph graph;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction similarityFunction;
  private final NodeScorerFactory scorerFactory;

  // Scratch buffers reused across searches — zero allocation per search.
  private final BitSet visited;
  private final NodeQueue candidates;
  private final NodeQueue results;
  // Scratch buffer for rescore(): avoids allocating a new NodeQueue on every two-pass call.
  private final NodeQueue rescoreHeap;
  // Scratch arrays for beamSearch() result reversal: pre-sized to graph capacity.
  private final int[] tmpNodes;
  private final float[] tmpScores;

  HnswSearcher(
      HnswGraph graph,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction,
      NodeScorerFactory scorerFactory) {
    this.graph = graph;
    this.vectors = vectors;
    this.similarityFunction = similarityFunction;
    this.scorerFactory = scorerFactory;
    int graphSize = Math.max(1, graph.size());
    this.visited = new BitSet(graphSize);
    this.candidates = new NodeQueue(256, false); // max-heap
    this.results = new NodeQueue(256, true); // min-heap
    this.rescoreHeap = new NodeQueue(64, true); // min-heap, grows as needed
    this.tmpNodes = new int[graphSize];
    this.tmpScores = new float[graphSize];
  }

  /** Creates a searcher using full-precision scoring. */
  HnswSearcher(
      HnswGraph graph, RandomAccessVectors vectors, SimilarityFunction similarityFunction) {
    this(
        graph,
        vectors,
        similarityFunction,
        query -> nodeId -> similarityFunction.compare(query, vectors.getVector(nodeId)));
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
    NeighborArray beamResults = beamSearch(new int[] {currentBest}, efSearch, 0, scorer);

    return extractTopK(beamResults, k);
  }

  /** Searches with default efSearch = max(k, 100). */
  public SearchResult search(float[] query, int k) {
    return search(query, k, Math.max(k, 100));
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

  /** Greedy descent: walk to the best neighbor at the given layer. */
  int greedyDescend(int entryPoint, int layer, NodeScorer scorer) {
    int current = entryPoint;
    float currentScore = scorer.score(current);

    boolean improved = true;
    while (improved) {
      improved = false;
      NeighborArray neighbors = graph.getNeighbors(current, layer);
      if (neighbors == null) break;

      for (int i = 0; i < neighbors.size(); i++) {
        int neighborId = neighbors.node(i);
        float score = scorer.score(neighborId);
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
  NeighborArray beamSearch(int[] entryPoints, int ef, int layer, NodeScorer scorer) {
    visited.clear();
    candidates.clear();
    results.clear();

    // Seed with entry points
    for (int ep : entryPoints) {
      if (!visited.get(ep)) {
        visited.set(ep);
        float score = scorer.score(ep);
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

        float score = scorer.score(neighborId);

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

    // Convert results min-heap to sorted NeighborArray (descending).
    // Reuse pre-allocated tmpNodes/tmpScores scratch fields — no allocation per search.
    int resultSize = results.size();
    var resultArray = new NeighborArray(resultSize == 0 ? 1 : resultSize);
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
