package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.quantization.ProductQuantizer;
import java.util.BitSet;

/**
 * HNSW searcher that uses Fused ADC scoring at layer 0.
 *
 * <p>At layer 0 (the dense navigation layer visited on every search), each neighbor's approximate
 * distance is computed via the precomputed ADC lookup table in O(M) additions — one per PQ
 * subspace. This is 3–10× faster than the full-precision O(dim) dot product used by {@link
 * HnswSearcher}, and eliminates the random-access hop to the global vector matrix because PQ codes
 * are stored adjacent to neighbor IDs in {@link FusedAdcNeighborList}.
 *
 * <p>Upper layers use exact scoring via a standard {@link NodeScorer} because they are visited far
 * less frequently (O(log n) total vs O(ef × degree) for layer 0).
 *
 * <p>Thread-safety: this class is <em>not thread-safe</em>; each search thread needs its own
 * instance. The {@link FusedAdcGraph} and query vectors are read-only.
 */
public final class FusedAdcHnswSearcher {

  private final FusedAdcGraph graph;
  private final float[][] vectors; // full-precision vectors for upper-layer exact scoring
  private final SimilarityFunction metric;
  private final ProductQuantizer pq;
  private final BitSet visited;
  private final NodeQueue candidates; // max-heap (minHeap=false): best candidate polled first
  private final NodeQueue results; // min-heap (minHeap=true): worst result polled first

  /**
   * Creates a searcher for the given fused ADC graph.
   *
   * @param graph the fused ADC graph to search
   * @param vectors the full-precision vector matrix (used for upper-layer exact scoring)
   * @param metric the similarity function
   * @param pq the trained PQ quantizer used to build ADC tables
   */
  public FusedAdcHnswSearcher(
      FusedAdcGraph graph, float[][] vectors, SimilarityFunction metric, ProductQuantizer pq) {
    this.graph = graph;
    this.vectors = vectors;
    this.metric = metric;
    this.pq = pq;
    this.visited = new BitSet(graph.size());
    this.candidates = new NodeQueue(Math.max(1, graph.size()), false); // max-heap
    this.results = new NodeQueue(Math.max(1, graph.size()), true); // min-heap
  }

  /**
   * Searches for the {@code k} approximate nearest neighbors using Fused ADC scoring.
   *
   * @param query the query vector (full precision)
   * @param k the number of results to return
   * @param efSearch the beam width (must be &ge; k; larger = better recall, slower)
   * @return the top-k search result
   */
  public SearchResult search(float[] query, int k, int efSearch) {
    boolean useDot =
        metric == SimilarityFunction.DOT_PRODUCT
            || metric == SimilarityFunction.MAXIMUM_INNER_PRODUCT
            || metric == SimilarityFunction.COSINE;

    // Precompute ADC lookup table once for this query.
    // table[m][k] = partial distance/similarity for subspace m and centroid k.
    // Raw sums are then converted to non-negative scores via toScore() so that
    // NodeQueue's non-negative float constraint is always satisfied.
    float[][] adcTable = pq.buildADCTable(query, useDot);

    // Upper-layer exact scorer for greedy descent (phase 1)
    NodeScorer exactScorer = nodeId -> metric.compare(query, vectors[nodeId]);

    int ep = graph.entryNode();
    int maxLevel = graph.maxLevel();

    // Phase 1: greedy exact descent from maxLevel to layer 1
    int currentBest = ep;
    for (int level = maxLevel; level >= 1; level--) {
      currentBest = greedyDescentExact(currentBest, level, exactScorer);
    }

    // Phase 2: fused ADC beam search at layer 0
    NeighborArray beamResults = fusedBeamSearch(new int[] {currentBest}, efSearch, adcTable);

    return extractTopK(beamResults, k);
  }

  /** Searches with default efSearch = max(k, 100). */
  public SearchResult search(float[] query, int k) {
    return search(query, k, Math.max(k, 100));
  }

  // --- Greedy descent (upper layers, exact scoring) ---

  private int greedyDescentExact(int entryPoint, int level, NodeScorer scorer) {
    int current = entryPoint;
    float currentScore = scorer.score(current);
    boolean improved = true;
    while (improved) {
      improved = false;
      NeighborArray neighbors = graph.upperNeighbors(current, level);
      if (neighbors == null) break;
      for (int i = 0; i < neighbors.size(); i++) {
        int nid = neighbors.node(i);
        float score = scorer.score(nid);
        if (score > currentScore) {
          current = nid;
          currentScore = score;
          improved = true;
        }
      }
    }
    return current;
  }

  // --- Fused ADC beam search (layer 0) ---

  private NeighborArray fusedBeamSearch(int[] entryPoints, int ef, float[][] adcTable) {
    visited.clear();
    candidates.clear();
    results.clear();

    for (int ep : entryPoints) {
      if (!visited.get(ep)) {
        visited.set(ep);
        float epScore = scoreEntry(ep, adcTable);
        candidates.add(ep, epScore);
        results.add(ep, epScore);
      }
    }

    while (!candidates.isEmpty()) {
      long top = candidates.poll();
      float topScore = NodeQueue.score(top);
      int topId = NodeQueue.nodeId(top);

      if (results.size() >= ef && topScore < NodeQueue.score(results.peek())) break;

      FusedAdcNeighborList nl = graph.fusedNeighbors(topId);
      if (nl == null) continue;

      for (int i = 0; i < nl.size(); i++) {
        int nid = nl.nodeId(i);
        if (visited.get(nid)) continue;
        visited.set(nid);
        float score = toScore(nl.adcScore(i, adcTable));

        if (results.size() < ef) {
          candidates.add(nid, score);
          results.add(nid, score);
        } else {
          float worst = NodeQueue.score(results.peek());
          if (score > worst) {
            candidates.add(nid, score);
            results.poll();
            results.add(nid, score);
          }
        }
      }
    }
    return drainResults();
  }

  private float scoreEntry(int nodeId, float[][] adcTable) {
    byte[] code = pq.encode(vectors[nodeId]);
    float sum = 0f;
    for (int m = 0; m < code.length; m++) {
      sum += adcTable[m][code[m] & 0xFF];
    }
    return toScore(sum);
  }

  /**
   * Converts a raw ADC partial sum to a non-negative similarity score using the same transform as
   * {@link SimilarityFunction#compare}. This satisfies {@link NodeQueue}'s non-negative constraint.
   *
   * <ul>
   *   <li>EUCLIDEAN: {@code 1 / (1 + adcSum)} — adcSum is sum of partial squared L2 distances
   *   <li>DOT_PRODUCT / COSINE: {@code (1 + adcSum) / 2} — adcSum is sum of partial dot products
   *   <li>MAXIMUM_INNER_PRODUCT: piecewise via {@link SimilarityFunction#scaleMaxInnerProductScore}
   * </ul>
   */
  private float toScore(float adcSum) {
    return switch (metric) {
      case EUCLIDEAN -> 1f / (1f + adcSum);
      case DOT_PRODUCT, COSINE -> (1f + adcSum) / 2f;
      case MAXIMUM_INNER_PRODUCT -> SimilarityFunction.scaleMaxInnerProductScore(adcSum);
    };
  }

  private NeighborArray drainResults() {
    // Min-heap polls in ascending score order; NeighborArray.insert() maintains descending order.
    int sz = results.size();
    NeighborArray out = new NeighborArray(sz);
    for (int i = 0; i < sz; i++) {
      long entry = results.poll();
      out.insert(NodeQueue.nodeId(entry), NodeQueue.score(entry));
    }
    return out;
  }

  private SearchResult extractTopK(NeighborArray beam, int k) {
    int sz = Math.min(beam.size(), k);
    int[] ids = new int[sz];
    float[] scores = new float[sz];
    for (int i = 0; i < sz; i++) {
      ids[i] = beam.node(i);
      scores[i] = beam.score(i);
    }
    return new SearchResult(ids, scores);
  }
}
