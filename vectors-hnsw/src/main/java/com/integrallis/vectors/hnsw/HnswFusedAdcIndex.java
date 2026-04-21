package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ProductQuantizer;

/**
 * HNSW index with Fused ADC (Asymmetric Distance Computation) scoring.
 *
 * <p>Fused ADC is the primary performance gap between java-vectors and JVector (see {@code
 * design-strategy.md §2.4}). JVector inlines transposed PQ codebooks into graph adjacency lists so
 * that accessing a neighbor's ID and its quantized representation fits in the same CPU cache line —
 * eliminating the random-access hop to the global vector matrix for every neighbor scored during
 * beam search.
 *
 * <p><b>Implementation approach:</b> The beam search algorithm is delegated to the proven {@link
 * HnswSearcher} via a custom {@link NodeScorerFactory} that computes ADC scores (O(M) table
 * lookups) instead of exact distances (O(dim) floating-point multiplications). Layer-0 neighbor
 * lists are additionally stored as {@link FusedAdcNeighborList} objects — the encoded PQ codes are
 * adjacent to the neighbor IDs in memory — enabling future optimisation of the scoring inner loop
 * for cache-line efficiency without changing the search algorithm.
 *
 * <p><b>Build process:</b>
 *
 * <ol>
 *   <li>Build the HNSW graph via the standard builder.
 *   <li>Train a {@link ProductQuantizer} on the corpus.
 *   <li>Encode all vectors to PQ byte codes.
 *   <li>Convert layer-0 neighbor lists to {@link FusedAdcNeighborList} (codes adjacent to IDs).
 * </ol>
 *
 * <p>Thread-safety: searches are thread-safe (one {@link HnswSearcher} per thread via {@link
 * ThreadLocal}).
 */
public final class HnswFusedAdcIndex {

  private final HnswGraph graph;
  private final InMemoryVectors ivec;
  private final SimilarityFunction metric;
  private final ProductQuantizer pq;
  private final byte[][] allCodes; // allCodes[nodeId] = M-byte PQ code
  private final FusedAdcGraph fusedGraph; // exposes fused layer-0 lists for inspection/future use
  private final ThreadLocal<HnswSearcher> searchers;

  private HnswFusedAdcIndex(
      HnswGraph graph,
      InMemoryVectors ivec,
      SimilarityFunction metric,
      ProductQuantizer pq,
      byte[][] allCodes,
      FusedAdcGraph fusedGraph) {
    this.graph = graph;
    this.ivec = ivec;
    this.metric = metric;
    this.pq = pq;
    this.allCodes = allCodes;
    this.fusedGraph = fusedGraph;
    // One HnswSearcher per thread, with an ADC NodeScorerFactory.
    this.searchers =
        ThreadLocal.withInitial(() -> new HnswSearcher(graph, ivec, metric, adcFactory()));
  }

  /**
   * Builds a {@link NodeScorerFactory} that uses the ADC lookup table for this index. Layer-0
   * neighbor expansion in beam search is served by {@link NodeScorer#scoreNeighborBatch}, which
   * reads the origin's packed codes in a single stride-1 SIMD sweep via {@link
   * com.integrallis.vectors.core.VectorUtil#batchAssembleAndSum}. Entry-point seeding, greedy
   * descent, and upper-layer scoring fall back to per-{@code nodeId} ADC lookups against {@link
   * #allCodes}.
   */
  private NodeScorerFactory adcFactory() {
    boolean useDot =
        metric == SimilarityFunction.DOT_PRODUCT
            || metric == SimilarityFunction.COSINE
            || metric == SimilarityFunction.MAXIMUM_INNER_PRODUCT;
    int M = pq.numSubspaces();
    return query -> {
      float[][] table = pq.buildADCTable(query, useDot);
      // Reusable scratch for the raw ADC sums before similarity remapping.
      float[] rawScratch = new float[graph.maxConnections0() + 2];
      return new NodeScorer() {
        @Override
        public float score(int nodeId) {
          return toScore(adcRawSum(table, allCodes[nodeId]), useDot);
        }

        @Override
        public boolean supportsNeighborBatch() {
          return true;
        }

        @Override
        public void scoreNeighborBatch(int originId, int count, float[] out) {
          FusedAdcNeighborList fused = fusedGraph.fusedNeighbors(originId);
          // Packed-layout batched ADC scan, then in-place similarity remap.
          com.integrallis.vectors.core.VectorUtil.batchAssembleAndSum(
              table, fused.packedCodes(), 0, rawScratch, count, M);
          for (int i = 0; i < count; i++) {
            out[i] = toScore(rawScratch[i], useDot);
          }
        }
      };
    };
  }

  private static float adcRawSum(float[][] table, byte[] codes) {
    float sum = 0f;
    for (int m = 0; m < codes.length; m++) sum += table[m][codes[m] & 0xFF];
    return sum;
  }

  private static float toScore(float adcSum, boolean useDot) {
    // Mirror the non-negative transforms in SimilarityFunction.compare() so that
    // NodeQueue's non-negative float constraint is always satisfied.
    if (useDot) return (1f + adcSum) / 2f;
    return 1f / (1f + adcSum); // EUCLIDEAN: adcSum = partial squared L2 distance sum >= 0
  }

  /**
   * Builds a {@link HnswFusedAdcIndex} from the given corpus.
   *
   * @param corpus full-precision corpus vectors
   * @param metric similarity function
   * @param maxConnections M (max neighbors per layer); 16 is a good default
   * @param efConstruction construction-time beam width; 200 is a good default
   * @param pqSubvectors number of PQ subspaces (must divide dim); 4–16 is typical
   * @param pqClusters clusters per subspace; 256 gives standard 8-bit codes
   * @param seed random seed
   */
  public static HnswFusedAdcIndex build(
      float[][] corpus,
      SimilarityFunction metric,
      int maxConnections,
      int efConstruction,
      int pqSubvectors,
      int pqClusters,
      long seed) {

    int n = corpus.length;
    InMemoryVectors ivec = new InMemoryVectors(corpus);

    // Step 1: Build the HNSW graph using the standard builder path.
    HnswGraph graph =
        HnswGraphBuilder.create(maxConnections, efConstruction, ivec, metric, seed).build();

    // Step 2: Train PQ quantizer on the corpus.
    ProductQuantizer pq =
        ProductQuantizer.train(new ArrayVectorDataset(corpus), pqSubvectors, pqClusters, true);

    // Step 3: Encode every vector to a PQ byte code.
    byte[][] allCodes = new byte[n][];
    for (int i = 0; i < n; i++) allCodes[i] = pq.encode(corpus[i]);

    // Step 4: Build fused layer-0 neighbor lists (neighbor codes packed contiguously per node).
    int M = pq.numSubspaces();
    FusedAdcNeighborList[] fused = new FusedAdcNeighborList[n];
    for (int nodeId = 0; nodeId < n; nodeId++) {
      NeighborArray neighbors = graph.getNeighbors(nodeId, 0);
      int sz = neighbors == null ? 0 : neighbors.size();
      int[] ids = new int[sz];
      for (int i = 0; i < sz; i++) ids[i] = neighbors.node(i);
      fused[nodeId] = FusedAdcNeighborList.pack(ids, allCodes, M);
    }

    return new HnswFusedAdcIndex(
        graph, ivec, metric, pq, allCodes, new FusedAdcGraph(graph, fused));
  }

  /**
   * Searches for the top-{@code k} approximate nearest neighbours using pure ADC scoring.
   *
   * <p>For production use at small-corpus or low-cluster-count PQ settings, prefer {@link
   * #searchTwoPass} which applies exact rescoring after ADC navigation.
   */
  public SearchResult search(float[] query, int k, int efSearch) {
    return searchers.get().search(query, k, efSearch);
  }

  /** Searches with default {@code efSearch = max(k, 100)}. */
  public SearchResult search(float[] query, int k) {
    return search(query, k, Math.max(k, 100));
  }

  /**
   * Two-pass search: ADC beam navigation (fast, approximate) followed by exact rescoring.
   *
   * <p>This is the recommended production path for Fused ADC. The ADC pass navigates to the
   * vicinity of the query using O(M) lookups per neighbor instead of O(dim) multiplications. The
   * exact rescore pass applies full-precision distance on the coarse candidates, recovering recall
   * that the ADC approximation may have lost.
   *
   * @param query the query vector (full precision)
   * @param k number of final results
   * @param efSearch beam width for the ADC coarse pass
   * @param overQueryFactor multiplier for coarse-pass k (e.g., 2.0 retrieves {@code 2k} candidates)
   * @return top-k results after exact rescoring, sorted by score descending
   */
  public SearchResult searchTwoPass(float[] query, int k, int efSearch, float overQueryFactor) {
    return searchers.get().searchTwoPass(query, k, efSearch, overQueryFactor);
  }

  /**
   * Two-pass search with default {@code efSearch = max(k, 100)} and {@code overQueryFactor = 2}.
   */
  public SearchResult searchTwoPass(float[] query, int k) {
    return searchTwoPass(query, k, Math.max(k, 100), 2.0f);
  }

  /** Number of indexed vectors. */
  public int size() {
    return allCodes.length;
  }

  /** Returns the underlying {@link ProductQuantizer} (for inspection or serialization). */
  public ProductQuantizer productQuantizer() {
    return pq;
  }

  /** Returns the fused layer-0 graph (exposes {@link FusedAdcNeighborList} per node). */
  public FusedAdcGraph fusedGraph() {
    return fusedGraph;
  }
}
