package com.integrallis.vectors.hnsw;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.BitSet;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for HNSW graph construction: level generation, neighbor selection, and insertion. */
class HnswGraphBuilderTest {

  @Nested
  @Tag("unit")
  class RandomLevelGeneratorTests {

    @Test
    void levelsFollowExponentialDistribution() {
      var gen = new RandomLevelGenerator(16, 42L);
      int[] counts = new int[10];
      int n = 100_000;
      for (int i = 0; i < n; i++) {
        int level = gen.nextLevel();
        if (level < counts.length) {
          counts[level]++;
        }
      }
      // ~93.75% at level 0 for M=16 (mL = 1/ln(16) ≈ 0.361)
      double ratio0 = (double) counts[0] / n;
      assertThat(ratio0).isBetween(0.92, 0.95);
      // Level 1 should be much less than level 0
      assertThat(counts[1]).isLessThan(counts[0]);
      // Level 2 should be much less than level 1
      assertThat(counts[2]).isLessThan(counts[1]);
    }

    @Test
    void deterministicWithSameSeed() {
      var gen1 = new RandomLevelGenerator(16, 42L);
      var gen2 = new RandomLevelGenerator(16, 42L);
      for (int i = 0; i < 100; i++) {
        assertThat(gen1.nextLevel()).isEqualTo(gen2.nextLevel());
      }
    }

    @Test
    void noNegativeLevels() {
      var gen = new RandomLevelGenerator(16, 42L);
      for (int i = 0; i < 10_000; i++) {
        assertThat(gen.nextLevel()).isGreaterThanOrEqualTo(0);
      }
    }

    @Test
    void mostNodesAtLevel0() {
      var gen = new RandomLevelGenerator(16, 42L);
      int level0Count = 0;
      int total = 10_000;
      for (int i = 0; i < total; i++) {
        if (gen.nextLevel() == 0) {
          level0Count++;
        }
      }
      assertThat(level0Count).isGreaterThan(total * 9 / 10);
    }
  }

  @Nested
  @Tag("unit")
  class NeighborSelectorTests {

    @Test
    void selectDiverse_returnsAtMostMaxConnections() {
      float[][] vectors = randomVectors(20, 8, 42L);
      var rav = new InMemoryVectors(vectors);
      var candidates = new NeighborArray(20);
      for (int i = 1; i < 20; i++) {
        float score = SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[i]);
        candidates.insert(i, score);
      }

      var result = NeighborSelector.selectDiverse(candidates, 4, rav, SimilarityFunction.EUCLIDEAN);
      assertThat(result.size()).isLessThanOrEqualTo(4);
    }

    @Test
    void selectDiverse_prefersCloserNeighbors() {
      // 3 vectors: query at origin, close at (0.1,...), far at (10,...)
      float[][] vectors = new float[3][4];
      vectors[1] = new float[] {0.1f, 0.1f, 0.1f, 0.1f};
      vectors[2] = new float[] {10f, 10f, 10f, 10f};

      var rav = new InMemoryVectors(vectors);
      var candidates = new NeighborArray(10);
      candidates.insert(1, SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[1]));
      candidates.insert(2, SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[2]));

      var result = NeighborSelector.selectDiverse(candidates, 2, rav, SimilarityFunction.EUCLIDEAN);
      assertThat(result.size()).isEqualTo(2);
      // Best neighbor (closest) should be first
      assertThat(result.node(0)).isEqualTo(1);
    }

    @Test
    void selectDiverse_prunesBlockedCandidates() {
      // Three colinear points: candidate B is between A and C, so B blocks C
      float[][] vectors = new float[4][2];
      vectors[0] = new float[] {0f, 0f}; // query
      vectors[1] = new float[] {1f, 0f}; // A (closest)
      vectors[2] = new float[] {2f, 0f}; // B (middle)
      vectors[3] = new float[] {3f, 0f}; // C (farthest)

      var rav = new InMemoryVectors(vectors);
      var candidates = new NeighborArray(10);
      candidates.insert(1, SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[1]));
      candidates.insert(2, SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[2]));
      candidates.insert(3, SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[3]));

      // With maxConn=1, only A should be selected (others blocked by A)
      var result = NeighborSelector.selectDiverse(candidates, 1, rav, SimilarityFunction.EUCLIDEAN);
      assertThat(result.size()).isEqualTo(1);
      assertThat(result.node(0)).isEqualTo(1);
    }

    @Test
    void selectDiverse_fillsWithPrunedWhenKeepPruned() {
      // With maxConnections=3, diverse selection + pruned fill to maxConn
      float[][] vectors = new float[5][2];
      vectors[0] = new float[] {0f, 0f}; // query
      vectors[1] = new float[] {1f, 0f};
      vectors[2] = new float[] {2f, 0f}; // blocked by 1
      vectors[3] = new float[] {0f, 1f}; // diverse direction
      vectors[4] = new float[] {3f, 0f}; // blocked by 1

      var rav = new InMemoryVectors(vectors);
      var candidates = new NeighborArray(10);
      for (int i = 1; i < 5; i++) {
        candidates.insert(i, SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[i]));
      }

      var result = NeighborSelector.selectDiverse(candidates, 3, rav, SimilarityFunction.EUCLIDEAN);
      assertThat(result.size()).isEqualTo(3);
    }

    @Test
    void selectDiverse_emptyCandidate_returnsEmpty() {
      float[][] vectors = new float[1][4];
      var rav = new InMemoryVectors(vectors);
      var candidates = new NeighborArray(10);

      var result = NeighborSelector.selectDiverse(candidates, 4, rav, SimilarityFunction.EUCLIDEAN);
      assertThat(result.size()).isZero();
    }

    @Test
    void selectDiverse_singleCandidate_returnsIt() {
      float[][] vectors = new float[2][4];
      vectors[1] = new float[] {1f, 1f, 1f, 1f};
      var rav = new InMemoryVectors(vectors);
      var candidates = new NeighborArray(10);
      candidates.insert(1, SimilarityFunction.EUCLIDEAN.compare(vectors[0], vectors[1]));

      var result = NeighborSelector.selectDiverse(candidates, 4, rav, SimilarityFunction.EUCLIDEAN);
      assertThat(result.size()).isEqualTo(1);
      assertThat(result.node(0)).isEqualTo(1);
    }
  }

  @Nested
  @Tag("unit")
  class SingleNodeInsertion {

    @Test
    void firstNode_becomesEntryPoint() {
      float[][] vectors = {{1f, 2f, 3f}};
      var builder =
          HnswGraphBuilder.create(
              16, 200, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      assertThat(graph.entryNode()).isEqualTo(0);
      assertThat(graph.size()).isEqualTo(1);
    }

    @Test
    void firstNode_hasNoNeighbors() {
      float[][] vectors = {{1f, 2f, 3f}};
      var builder =
          HnswGraphBuilder.create(
              16, 200, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      assertThat(graph.getNeighbors(0, 0).size()).isZero();
    }

    @Test
    void secondNode_hasEdgeToFirst() {
      float[][] vectors = {{1f, 2f, 3f}, {4f, 5f, 6f}};
      var builder =
          HnswGraphBuilder.create(
              16, 200, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      assertThat(graph.getNeighbors(1, 0).contains(0)).isTrue();
    }

    @Test
    void secondNode_firstHasEdgeToSecond() {
      float[][] vectors = {{1f, 2f, 3f}, {4f, 5f, 6f}};
      var builder =
          HnswGraphBuilder.create(
              16, 200, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      // Bidirectional: first node should also have edge to second
      assertThat(graph.getNeighbors(0, 0).contains(1)).isTrue();
    }
  }

  @Nested
  @Tag("unit")
  class SmallGraphConstruction {

    @Test
    void tenNodes_allReachableFromEntryPoint() {
      float[][] vectors = randomVectors(10, 8, 42L);
      var builder =
          HnswGraphBuilder.create(
              4, 50, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      // BFS from entry point should reach all nodes
      BitSet visited = new BitSet(10);
      bfs(graph, graph.entryNode(), 0, visited);
      assertThat(visited.cardinality()).isEqualTo(10);
    }

    @Test
    void tenNodes_entryPointIsHighestLevelNode() {
      float[][] vectors = randomVectors(10, 8, 42L);
      var builder =
          HnswGraphBuilder.create(
              4, 50, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      int ep = graph.entryNode();
      int epLevel = graph.nodeLevel(ep);
      assertThat(epLevel).isEqualTo(graph.maxLevel());
    }

    @Test
    void tenNodes_neighborCountsWithinLimits() {
      float[][] vectors = randomVectors(10, 8, 42L);
      var builder =
          HnswGraphBuilder.create(
              4, 50, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      for (int i = 0; i < 10; i++) {
        var neighbors = graph.getNeighbors(i, 0);
        assertThat(neighbors.size()).isLessThanOrEqualTo(graph.maxConnections0());
      }
    }

    @Test
    void tenNodes_mostEdgesAreBidirectional() {
      float[][] vectors = randomVectors(10, 8, 42L);
      var builder =
          HnswGraphBuilder.create(
              4, 50, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      // Pruning can break strict bidirectionality, but most edges should be bidirectional
      int totalEdges = 0;
      int bidirectionalEdges = 0;
      for (int i = 0; i < 10; i++) {
        var neighbors = graph.getNeighbors(i, 0);
        for (int j = 0; j < neighbors.size(); j++) {
          totalEdges++;
          int neighbor = neighbors.node(j);
          if (graph.getNeighbors(neighbor, 0).contains(i)) {
            bidirectionalEdges++;
          }
        }
      }
      double ratio = (double) bidirectionalEdges / totalEdges;
      assertThat(ratio)
          .as(
              "At least 80%% of edges should be bidirectional, got %d/%d",
              bidirectionalEdges, totalEdges)
          .isGreaterThan(0.80);
    }
  }

  @Nested
  @Tag("unit")
  class MediumGraphConstruction {

    @Test
    void hundredNodes_graphIsConnected() {
      float[][] vectors = randomVectors(100, 32, 42L);
      var builder =
          HnswGraphBuilder.create(
              16, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      BitSet visited = new BitSet(100);
      bfs(graph, graph.entryNode(), 0, visited);
      assertThat(visited.cardinality()).isEqualTo(100);
    }

    @Test
    void hundredNodes_deterministicWithSameSeed() {
      float[][] vectors = randomVectors(100, 32, 42L);

      var builder1 =
          HnswGraphBuilder.create(
              16, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 99L);
      var graph1 = builder1.build();

      var builder2 =
          HnswGraphBuilder.create(
              16, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 99L);
      var graph2 = builder2.build();

      assertThat(graph1.entryNode()).isEqualTo(graph2.entryNode());
      assertThat(graph1.maxLevel()).isEqualTo(graph2.maxLevel());
      assertThat(graph1.size()).isEqualTo(graph2.size());
    }

    @Test
    void hundredNodes_neighborDiversity() {
      float[][] vectors = randomVectors(100, 32, 42L);
      var builder =
          HnswGraphBuilder.create(
              16, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L);
      var graph = builder.build();

      // Average neighbor count at layer 0 should be reasonable (> 2 per node)
      int totalNeighbors = 0;
      for (int i = 0; i < 100; i++) {
        totalNeighbors += graph.getNeighbors(i, 0).size();
      }
      double avgNeighbors = (double) totalNeighbors / 100;
      assertThat(avgNeighbors).isGreaterThan(2.0);
    }
  }

  @Nested
  @Tag("unit")
  class ConcurrentBuild {

    @Test
    void concurrentBuild_mainComponentAtLeast95pct() {
      // Concurrent HNSW cannot guarantee 100% connectivity for small n (same limitation
      // as hnswlib / JVector): early nodes inserted simultaneously all see only the entry
      // node and may form a thin star rather than a fully-woven mesh.  95% is a realistic
      // lower bound for n=200, M=8, 4 threads.
      float[][] vectors = randomVectors(200, 16, 42L);
      var graph =
          ConcurrentHnswGraphBuilder.create(
                  8, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L)
              .build(4);

      BitSet visited = new BitSet(200);
      bfs(graph, graph.entryNode(), 0, visited);
      int reachable = visited.cardinality();
      assertThat(reachable)
          .as(
              "Main connected component should contain >= 95%% of nodes (%d/200 reachable)",
              reachable)
          .isGreaterThanOrEqualTo(190);
    }

    @Test
    void concurrentBuild_noIsolatedNodes() {
      float[][] vectors = randomVectors(200, 16, 42L);
      var graph =
          ConcurrentHnswGraphBuilder.create(
                  8, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L)
              .build(4);

      for (int i = 0; i < 200; i++) {
        assertThat(graph.getNeighbors(i, 0).size())
            .as("Node %d should have at least 1 neighbor", i)
            .isGreaterThan(0);
      }
    }

    @Test
    void concurrentBuild_neighborCountsWithinLimits() {
      float[][] vectors = randomVectors(200, 16, 42L);
      var graph =
          ConcurrentHnswGraphBuilder.create(
                  8, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L)
              .build(4);

      for (int i = 0; i < 200; i++) {
        assertThat(graph.getNeighbors(i, 0).size())
            .as("Node %d layer-0 neighbor count", i)
            .isLessThanOrEqualTo(graph.maxConnections0());
      }
    }

    @Test
    void concurrentBuild_achievesGoodRecall() {
      // Recall gate: concurrent HNSW recall@5 vs brute-force >= 0.80 with n=500
      int n = 500;
      int dim = 32;
      int k = 5;
      float[][] vecs = randomVectors(n, dim, 7L);
      float[][] queries = randomVectors(10, dim, 77L);

      var graph =
          ConcurrentHnswGraphBuilder.create(
                  16, 200, new InMemoryVectors(vecs), SimilarityFunction.EUCLIDEAN, 42L)
              .build(4);
      var searcher =
          new HnswSearcher(graph, new InMemoryVectors(vecs), SimilarityFunction.EUCLIDEAN);

      int hits = 0, total = 0;
      for (float[] q : queries) {
        // Brute-force top-k
        var bruteHeap = new NodeQueue(n, true);
        for (int i = 0; i < n; i++) {
          float s = SimilarityFunction.EUCLIDEAN.compare(q, vecs[i]);
          if (bruteHeap.size() < k) bruteHeap.add(i, s);
          else if (s > NodeQueue.score(bruteHeap.peek())) {
            bruteHeap.poll();
            bruteHeap.add(i, s);
          }
        }
        java.util.Set<Integer> gt = new java.util.HashSet<>();
        while (!bruteHeap.isEmpty()) gt.add(NodeQueue.nodeId(bruteHeap.poll()));

        var result = searcher.search(q, k, 200);
        for (int i = 0; i < result.size(); i++) if (gt.contains(result.nodeId(i))) hits++;
        total += gt.size();
      }

      double recall = (double) hits / total;
      assertThat(recall)
          .as("Concurrent HNSW recall@5 vs brute-force should be >= 0.80, was %.3f", recall)
          .isGreaterThanOrEqualTo(0.80);
    }
  }

  // --- Helpers ---

  static float[][] randomVectors(int count, int dimension, long seed) {
    var rng = new Random(seed);
    float[][] vectors = new float[count][dimension];
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < dimension; j++) {
        vectors[i][j] = rng.nextFloat();
      }
    }
    return vectors;
  }

  /** BFS traversal from a start node at the given layer, marking visited nodes. */
  static void bfs(HnswGraph graph, int startNode, int layer, BitSet visited) {
    var queue = new java.util.ArrayDeque<Integer>();
    queue.add(startNode);
    visited.set(startNode);
    while (!queue.isEmpty()) {
      int node = queue.poll();
      var neighbors = graph.getNeighbors(node, layer);
      if (neighbors == null) continue;
      for (int i = 0; i < neighbors.size(); i++) {
        int neighbor = neighbors.node(i);
        if (!visited.get(neighbor)) {
          visited.set(neighbor);
          queue.add(neighbor);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // SSD prefetch gate tests (DP5)
  // ---------------------------------------------------------------------------

  @Nested
  @Tag("unit")
  class SsdPrefetchTests {

    private static final int N = 1_000;
    private static final int DIM = 32;

    private static float[][] randomVecs(int n, int dim, long seed) {
      Random rng = new Random(seed);
      float[][] v = new float[n][dim];
      for (float[] row : v) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat();
      return v;
    }

    private static int[] bruteTopK(float[] q, float[][] data, int k) {
      int n = data.length;
      Integer[] idx = new Integer[n];
      float[] dists = new float[n];
      for (int i = 0; i < n; i++) {
        idx[i] = i;
        float s = 0f;
        for (int d = 0; d < q.length; d++) {
          float diff = q[d] - data[i][d];
          s += diff * diff;
        }
        dists[i] = s;
      }
      java.util.Arrays.sort(idx, (a, b) -> Float.compare(dists[a], dists[b]));
      int[] result = new int[k];
      for (int i = 0; i < k; i++) result[i] = idx[i];
      return result;
    }

    private static HnswIndex buildIndex(float[][] data, int m, int ef, long seed) {
      return HnswIndex.builder(data, SimilarityFunction.EUCLIDEAN)
          .maxConnections(m)
          .efConstruction(ef)
          .seed(seed)
          .build();
    }

    @Test
    void searchWithPrefetch_producesIdenticalResultsToNormalSearch() {
      float[][] data = randomVecs(N, DIM, 1L);
      HnswIndex idx = buildIndex(data, 16, 200, 42L);
      float[] query = randomVecs(1, DIM, 99L)[0];
      int k = 10, ef = 50;

      SearchResult baseline = idx.search(query, k, ef);

      try (AsyncVectorPrefetcher prefetcher = new AsyncVectorPrefetcher(idx.vectorSource(), 2)) {
        SearchResult withPrefetch = idx.searchWithPrefetch(query, k, ef, prefetcher);
        assertThat(withPrefetch.nodeIds()).containsExactly(baseline.nodeIds());
      }
    }

    @Test
    void asyncVectorPrefetcher_submittedCount_incrementsPerPrefetchCall() {
      float[][] data = randomVecs(100, DIM, 2L);
      HnswIndex idx = buildIndex(data, 8, 100, 42L);

      try (AsyncVectorPrefetcher prefetcher = new AsyncVectorPrefetcher(idx.vectorSource(), 2)) {
        assertThat(prefetcher.submittedCount()).isEqualTo(0);
        prefetcher.prefetch(0);
        prefetcher.prefetch(1);
        prefetcher.prefetch(2);
        assertThat(prefetcher.submittedCount()).isEqualTo(3);
      }
    }

    @Test
    void asyncVectorPrefetcher_outOfRangeOrdinal_isIgnoredSilently() {
      float[][] data = randomVecs(10, DIM, 3L);
      HnswIndex idx = buildIndex(data, 4, 50, 42L);

      try (AsyncVectorPrefetcher prefetcher = new AsyncVectorPrefetcher(idx.vectorSource(), 1)) {
        assertThatNoException()
            .isThrownBy(
                () -> {
                  prefetcher.prefetch(-1);
                  prefetcher.prefetch(10_000);
                });
        // Only valid ordinals counted
        assertThat(prefetcher.submittedCount()).isEqualTo(0);
      }
    }

    @Test
    void searchWithPrefetch_recall_atLeast90Pct_vs_bruteForce() {
      float[][] data = randomVecs(N, DIM, 4L);
      HnswIndex idx = buildIndex(data, 16, 200, 42L);
      int k = 10, ef = 50, queries = 30;
      double totalRecall = 0;

      try (AsyncVectorPrefetcher prefetcher = new AsyncVectorPrefetcher(idx.vectorSource(), 2)) {
        for (int q = 0; q < queries; q++) {
          float[] query = randomVecs(1, DIM, 500L + q)[0];
          int[] gt = bruteTopK(query, data, k);
          SearchResult r = idx.searchWithPrefetch(query, k, ef, prefetcher);
          int[] found = r.nodeIds();
          int hits = 0;
          for (int f : found)
            for (int g : gt)
              if (f == g) {
                hits++;
                break;
              }
          totalRecall += (double) hits / k;
        }
      }
      assertThat(totalRecall / queries).isGreaterThanOrEqualTo(0.90);
    }

    @Test
    void ssdHnswConfig_defaults_valid() {
      SsdHnswConfig cfg = SsdHnswConfig.defaults();
      assertThat(cfg.ioThreads()).isGreaterThanOrEqualTo(1);
      assertThat(cfg.prefetchWindowSize()).isGreaterThanOrEqualTo(1);
    }
  }
}
