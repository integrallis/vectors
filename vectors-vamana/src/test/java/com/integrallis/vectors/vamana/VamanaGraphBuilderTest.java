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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.ArrayDeque;
import java.util.BitSet;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class VamanaGraphBuilderTest {

  @Nested
  @Tag("unit")
  class MedoidComputation {

    @Test
    void medoidIsClosestToCentroid() {
      // Points: centroid is (2, 2). Point 2 at (2, 2) is the medoid.
      float[][] data = {{0, 0}, {4, 4}, {2, 2}, {0, 4}};
      var vectors = new InMemoryVectors(data);
      var builder =
          VamanaGraphBuilder.create(2, 10, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L);
      int medoid = builder.computeMedoid();

      // Point (2,2) is closest to centroid (1.5, 2.5)
      // Centroid = (0+4+2+0)/4=1.5, (0+4+2+4)/4=2.5
      // Distances: (0,0)→(1.5,2.5)=2.92, (4,4)→=2.92, (2,2)→=0.71, (0,4)→=2.12
      assertThat(medoid).isEqualTo(2);
    }

    @Test
    void singleVector_isMedoid() {
      float[][] data = {{1, 2, 3}};
      var vectors = new InMemoryVectors(data);
      var builder =
          VamanaGraphBuilder.create(2, 10, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L);
      assertThat(builder.computeMedoid()).isEqualTo(0);
    }

    @Test
    void medoidUsesL2Geometry_forNonEuclideanSimilarity() {
      // Points: centroid = (1.5, 2.5); L2-nearest = index 2 at (2, 2) (dist ≈ 0.71).
      // DOT_PRODUCT-maximiser of the centroid: (0,0)·c=0, (4,4)·c=16, (2,2)·c=8, (0,4)·c=10
      // → index 1 at (4,4) has the highest dot product with the centroid, but is NOT the medoid.
      // The medoid must always be the L2-nearest point to the centroid regardless of sim.
      float[][] data = {{0, 0}, {4, 4}, {2, 2}, {0, 4}};
      var vectors = new InMemoryVectors(data);
      var builder =
          VamanaGraphBuilder.create(2, 10, 1.2f, vectors, SimilarityFunction.DOT_PRODUCT, 42L);
      assertThat(builder.computeMedoid())
          .as(
              "medoid should be the L2-nearest to centroid (index 2), not the dot-product maximiser")
          .isEqualTo(2);
    }
  }

  @Nested
  @Tag("unit")
  class RandomInitialization {

    @Test
    void allNodesHaveNeighbors() {
      float[][] data = generateRandomVectors(20, 8, 42L);
      var vectors = new InMemoryVectors(data);
      VamanaGraph graph =
          VamanaGraphBuilder.create(4, 20, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build();

      for (int i = 0; i < graph.size(); i++) {
        assertThat(graph.getNeighbors(i).size())
            .as("Node %d should have neighbors", i)
            .isGreaterThan(0);
      }
    }

    @Test
    void noSelfLoops() {
      float[][] data = generateRandomVectors(20, 8, 42L);
      var vectors = new InMemoryVectors(data);
      VamanaGraph graph =
          VamanaGraphBuilder.create(4, 20, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build();

      for (int i = 0; i < graph.size(); i++) {
        var neighbors = graph.getNeighbors(i);
        for (int j = 0; j < neighbors.size(); j++) {
          assertThat(neighbors.node(j)).as("Node %d should not have self-loop", i).isNotEqualTo(i);
        }
      }
    }
  }

  @Nested
  @Tag("unit")
  class SmallGraphConstruction {

    @Test
    void fiveNodes_allReachableFromMedoid() {
      float[][] data = generateRandomVectors(5, 4, 42L);
      var vectors = new InMemoryVectors(data);
      VamanaGraph graph =
          VamanaGraphBuilder.create(3, 10, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build();

      // BFS from medoid should reach all nodes
      BitSet reached = bfsFrom(graph, graph.medoid());
      assertThat(reached.cardinality()).isEqualTo(5);
    }

    @Test
    void tenNodes_neighborCountsWithinR() {
      float[][] data = generateRandomVectors(10, 8, 42L);
      var vectors = new InMemoryVectors(data);
      int R = 4;
      VamanaGraph graph =
          VamanaGraphBuilder.create(R, 20, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build();

      for (int i = 0; i < graph.size(); i++) {
        assertThat(graph.getNeighbors(i).size())
            .as("Node %d neighbor count", i)
            .isLessThanOrEqualTo(R);
      }
    }

    @Test
    void tenNodes_mostEdgesAreBidirectional() {
      float[][] data = generateRandomVectors(10, 8, 42L);
      var vectors = new InMemoryVectors(data);
      VamanaGraph graph =
          VamanaGraphBuilder.create(4, 20, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build();

      int totalEdges = 0;
      int bidirectional = 0;
      for (int i = 0; i < graph.size(); i++) {
        var neighbors = graph.getNeighbors(i);
        for (int j = 0; j < neighbors.size(); j++) {
          totalEdges++;
          int neighbor = neighbors.node(j);
          if (graph.getNeighbors(neighbor).contains(i)) {
            bidirectional++;
          }
        }
      }
      // At least 60% of edges should be bidirectional (pruning can break some)
      double ratio = (double) bidirectional / totalEdges;
      assertThat(ratio)
          .as("Bidirectional edge ratio should be >= 0.60")
          .isGreaterThanOrEqualTo(0.60);
    }
  }

  @Nested
  @Tag("unit")
  class MediumGraphConstruction {

    @Test
    void hundredNodes_allReachableFromMedoid() {
      float[][] data = generateRandomVectors(100, 16, 42L);
      var vectors = new InMemoryVectors(data);
      VamanaGraph graph =
          VamanaGraphBuilder.create(16, 50, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build();

      BitSet reached = bfsFrom(graph, graph.medoid());
      assertThat(reached.cardinality()).isEqualTo(100);
    }

    @Test
    void hundredNodes_deterministicWithSameSeed() {
      float[][] data = generateRandomVectors(100, 16, 42L);
      var vectors = new InMemoryVectors(data);

      VamanaGraph graph1 =
          VamanaGraphBuilder.create(16, 50, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 99L)
              .build();
      VamanaGraph graph2 =
          VamanaGraphBuilder.create(16, 50, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 99L)
              .build();

      assertThat(graph1.medoid()).isEqualTo(graph2.medoid());
      for (int i = 0; i < graph1.size(); i++) {
        var n1 = graph1.getNeighbors(i);
        var n2 = graph2.getNeighbors(i);
        assertThat(n1.size()).as("Node %d neighbor count", i).isEqualTo(n2.size());
        for (int j = 0; j < n1.size(); j++) {
          assertThat(n1.node(j)).isEqualTo(n2.node(j));
        }
      }
    }
  }

  @Nested
  @Tag("unit")
  class ConcurrentBuild {

    @Test
    void concurrentBuild_graphIsConnected() {
      float[][] data = generateRandomVectors(200, 16, 42L);
      var vectors = new InMemoryVectors(data);
      VamanaGraph graph =
          ConcurrentVamanaGraphBuilder.create(
                  16, 50, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build(4);

      BitSet reached = bfsFrom(graph, graph.medoid());
      assertThat(reached.cardinality())
          .as("All nodes reachable from medoid in concurrent graph")
          .isEqualTo(200);
    }

    @Test
    void concurrentBuild_neighborCountsWithinR() {
      float[][] data = generateRandomVectors(200, 16, 42L);
      var vectors = new InMemoryVectors(data);
      int R = 16;
      VamanaGraph graph =
          ConcurrentVamanaGraphBuilder.create(
                  R, 50, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build(4);

      for (int i = 0; i < graph.size(); i++) {
        assertThat(graph.getNeighbors(i).size())
            .as("Node %d neighbor count", i)
            .isLessThanOrEqualTo(R);
      }
    }

    @Test
    void concurrentBuild_achievesGoodRecall() {
      // Recall gate: concurrent Vamana recall@5 vs brute-force >= 0.90 with n=500
      int n = 500;
      int dim = 32;
      int k = 5;
      float[][] data = generateRandomVectors(n, dim, 7L);
      var vectors = new InMemoryVectors(data);
      VamanaGraph graph =
          ConcurrentVamanaGraphBuilder.create(
                  16, 50, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
              .build(4);
      var searcher = new VamanaSearcher(graph, vectors, SimilarityFunction.EUCLIDEAN);

      int hits = 0, total = 0;
      for (float[] q : generateRandomVectors(10, dim, 77L)) {
        // Brute-force top-k
        var bruteHeap = new NodeQueue(n, true);
        for (int i = 0; i < n; i++) {
          float s = SimilarityFunction.EUCLIDEAN.compare(q, data[i]);
          if (bruteHeap.size() < k) bruteHeap.add(i, s);
          else if (s > NodeQueue.score(bruteHeap.peek())) {
            bruteHeap.poll();
            bruteHeap.add(i, s);
          }
        }
        java.util.Set<Integer> gt = new java.util.HashSet<>();
        while (!bruteHeap.isEmpty()) gt.add(NodeQueue.nodeId(bruteHeap.poll()));

        SearchResult result = searcher.search(q, k, 100);
        for (int i = 0; i < result.size(); i++) if (gt.contains(result.nodeId(i))) hits++;
        total += gt.size();
      }

      double recall = (double) hits / total;
      assertThat(recall)
          .as("Concurrent Vamana recall@5 vs brute-force should be >= 0.90, was %.3f", recall)
          .isGreaterThanOrEqualTo(0.90);
    }

    @Test
    void concurrentBuild_stressDifferentThreadCounts() {
      // Verify that concurrent build produces a valid, searchable graph regardless of thread count.
      float[][] data = generateRandomVectors(300, 16, 99L);
      float[] query = generateRandomVectors(1, 16, 100L)[0];

      for (int threads : new int[] {1, 2, 4, 8}) {
        var vectors = new InMemoryVectors(data);
        VamanaGraph graph =
            ConcurrentVamanaGraphBuilder.create(
                    16, 50, 1.2f, vectors, SimilarityFunction.EUCLIDEAN, 42L)
                .build(threads);

        // Graph must have all nodes
        assertThat(graph.size()).as("Graph size with %d threads", threads).isEqualTo(300);

        // All nodes must have at least one neighbor
        for (int i = 0; i < 300; i++) {
          assertThat(graph.getNeighbors(i).size())
              .as("Node %d with %d threads", i, threads)
              .isGreaterThan(0);
        }

        // Graph must be searchable and return valid results
        var searcher = new VamanaSearcher(graph, vectors, SimilarityFunction.EUCLIDEAN);
        SearchResult result = searcher.search(query, 10, 100);
        assertThat(result.size()).as("Search results with %d threads", threads).isEqualTo(10);

        // All result scores must be finite
        for (int i = 0; i < result.size(); i++) {
          assertThat(Float.isFinite(result.score(i)))
              .as("Score %d finite with %d threads", i, threads)
              .isTrue();
        }
      }
    }
  }

  // --- Helpers ---

  /** Canonical random-vector generator shared by all Vamana test classes (package-private). */
  static float[][] generateRandomVectors(int count, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] data = new float[count][dim];
    for (int i = 0; i < count; i++) {
      for (int d = 0; d < dim; d++) {
        data[i][d] = rng.nextFloat();
      }
    }
    return data;
  }

  private static BitSet bfsFrom(VamanaGraph graph, int start) {
    BitSet visited = new BitSet(graph.size());
    var queue = new ArrayDeque<Integer>();
    queue.add(start);
    visited.set(start);
    while (!queue.isEmpty()) {
      int node = queue.poll();
      var neighbors = graph.getNeighbors(node);
      for (int i = 0; i < neighbors.size(); i++) {
        int neighbor = neighbors.node(i);
        if (!visited.get(neighbor)) {
          visited.set(neighbor);
          queue.add(neighbor);
        }
      }
    }
    return visited;
  }
}
