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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.SplittableRandom;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
@DisplayName("HnswGraphDiagnostics")
class HnswGraphDiagnosticsTest {

  @Nested
  @DisplayName("hand-built graphs")
  class HandBuilt {

    @Test
    @DisplayName("an empty graph reports nothing and does not throw")
    void emptyGraph() {
      HnswGraphDiagnostics.Report report = HnswGraphDiagnostics.inspect(new HnswGraph(8, 4));
      assertThat(report.nodes()).isZero();
      assertThat(report.isFullyReachable()).isTrue();
      assertThat(report.isHealthy()).isTrue();
    }

    @Test
    @DisplayName("detects a node unreachable from the entry node")
    void detectsUnreachableNode() {
      // Two nodes, no edges between them. Node 1 exists but greedy search starting at node 0 can
      // never arrive, so it can never be returned at any efSearch -- the exact failure an average
      // recall number hides.
      HnswGraph graph = new HnswGraph(2, 4);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.setEntryNode(0, 0);

      HnswGraphDiagnostics.Report report = HnswGraphDiagnostics.inspect(graph);
      assertThat(report.nodes()).isEqualTo(2);
      assertThat(report.unreachableFromEntry()).isEqualTo(1);
      assertThat(report.isFullyReachable()).isFalse();
      assertThat(report.isolatedNodes()).isEqualTo(2);
      assertThat(report.isHealthy()).isFalse();
    }

    @Test
    @DisplayName("follows directed edges only, so a one-way edge leaves the source unreachable")
    void reachabilityIsDirected() {
      // 0 -> 1 but not 1 -> 0. Entry is 1, so 0 cannot be reached. Treating the graph as undirected
      // would wrongly call this healthy, and greedy search is directed.
      HnswGraph graph = new HnswGraph(2, 4);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.getNeighbors(0, 0).insert(1, 0.9f);
      graph.setEntryNode(1, 0);

      assertThat(HnswGraphDiagnostics.inspect(graph).unreachableFromEntry()).isEqualTo(1);
    }

    @Test
    @DisplayName("counts self loops")
    void countsSelfLoops() {
      HnswGraph graph = new HnswGraph(2, 4);
      graph.initNode(0, 0);
      graph.initNode(1, 0);
      graph.getNeighbors(0, 0).insert(0, 1.0f);
      graph.getNeighbors(0, 0).insert(1, 0.9f);
      graph.setEntryNode(0, 0);

      HnswGraphDiagnostics.Report report = HnswGraphDiagnostics.inspect(graph);
      assertThat(report.selfLoops()).isEqualTo(1);
      assertThat(report.isHealthy()).isFalse();
      assertThat(report.unreachableFromEntry()).isZero();
    }

    @Test
    @DisplayName("reports degree statistics over layer 0")
    void reportsDegreeStats() {
      HnswGraph graph = new HnswGraph(3, 4);
      for (int i = 0; i < 3; i++) {
        graph.initNode(i, 0);
      }
      graph.getNeighbors(0, 0).insert(1, 0.9f);
      graph.getNeighbors(0, 0).insert(2, 0.8f);
      graph.getNeighbors(1, 0).insert(0, 0.9f);
      graph.setEntryNode(0, 0);

      HnswGraphDiagnostics.Report report = HnswGraphDiagnostics.inspect(graph);
      assertThat(report.minOutDegree()).isZero(); // node 2 has no out-edges
      assertThat(report.maxOutDegree()).isEqualTo(2);
      assertThat(report.meanOutDegree()).isEqualTo(1.0);
      assertThat(report.isolatedNodes()).isEqualTo(1);
      // Node 2 is still reachable: node 0 points at it even though it points back at nobody.
      assertThat(report.unreachableFromEntry()).isZero();
    }
  }

  @Nested
  @DisplayName("real builds")
  class RealBuilds {

    @Test
    @DisplayName("a normally built graph is fully reachable with no structural defects")
    void builtGraphIsHealthy() {
      float[][] vectors = randomUnitVectors(2_000, 32, 7L);
      HnswIndex index =
          HnswIndex.builder(new InMemoryVectors(vectors), SimilarityFunction.COSINE)
              .maxConnections(16)
              .efConstruction(100)
              .seed(11L)
              .build();

      HnswGraphDiagnostics.Report report = HnswGraphDiagnostics.inspect(index.graph());
      assertThat(report.nodes()).isEqualTo(2_000);
      // A serial build at this size reaches every node. That is not true at every size: measured on
      // fashion-mnist at 60,000 nodes the same configuration leaves 341 (0.57%) unreachable at
      // layer zero. Upper-layer routes must also be checked before inferring a recall ceiling.
      // The property being pinned here is that
      // the serial builder is clean at small scale, which makes the parallel contrast below real
      // rather than an artifact of the dataset.
      assertThat(report.unreachableFromEntry())
          .as("every node must be reachable from the entry node: %s", report)
          .isZero();
      assertThat(report.danglingEdges()).isZero();
      assertThat(report.selfLoops()).isZero();
      assertThat(report.isolatedNodes()).isZero();
      assertThat(report.isHealthy()).isTrue();
    }

    @Test
    @DisplayName("a parallel build loses some reachability, and this bounds how much")
    void parallelBuildReachabilityIsBounded() {
      // Measured 2026-10-02, not assumed. At this size a serial build leaves every node reachable
      // (see the test above) while a 4-thread build of the same data leaves 2 of 2,000 unreachable
      // and drops one node to out-degree 1. Concurrent inserts beam-search through nodes that have
      // been allocated but not yet inserted, so an early insert can find few live candidates; the
      // serial builder cannot hit this because node i only ever searches nodes 0..i-1.
      //
      // Asserted as a bound rather than relaxed to "anything goes": the number is small and this
      // pins it, so a change that makes concurrent builds materially worse fails here. The
      // published
      // remedy -- pre-linking the in-flight batch before running the rest of the insert -- is
      // recorded in research/HNSW-OVERLOOKED-TECHNIQUES.md and not yet implemented.
      int nodes = 2_000;
      float[][] vectors = randomUnitVectors(nodes, 32, 7L);
      HnswIndex index =
          HnswIndex.builder(new InMemoryVectors(vectors), SimilarityFunction.COSINE)
              .maxConnections(16)
              .efConstruction(100)
              .seed(11L)
              .parallelism(4)
              .build();

      HnswGraphDiagnostics.Report report = HnswGraphDiagnostics.inspect(index.graph());
      assertThat(report.unreachableFromEntry())
          .as("parallel build reachability regressed well beyond the measured bound: %s", report)
          .isLessThanOrEqualTo(nodes / 100); // 1%, against a measured 0.1%
      assertThat(report.danglingEdges()).isZero();
      assertThat(report.selfLoops()).isZero();
      assertThat(report.isolatedNodes()).isZero();
    }

    @Test
    @DisplayName("identical vectors, the degenerate case that broke Lucene's component repair")
    void identicalVectorsStillReachable() {
      // Lucene reverted automatic component-joining because it took an inordinate amount of time on
      // exactly this input. We only diagnose, so this must terminate quickly and report honestly
      // whatever the builder produced.
      float[][] vectors = new float[500][16];
      for (float[] v : vectors) {
        v[0] = 1f;
      }
      HnswIndex index =
          HnswIndex.builder(new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN)
              .maxConnections(8)
              .efConstruction(40)
              .seed(3L)
              .build();

      HnswGraphDiagnostics.Report report = HnswGraphDiagnostics.inspect(index.graph());
      assertThat(report.nodes()).isEqualTo(500);
      assertThat(report.danglingEdges()).isZero();
      assertThat(report.selfLoops()).isZero();
    }
  }

  private static float[][] randomUnitVectors(int count, int dimension, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][] out = new float[count][dimension];
    for (int i = 0; i < count; i++) {
      double sum = 0;
      for (int d = 0; d < dimension; d++) {
        out[i][d] = (float) random.nextGaussian();
        sum += (double) out[i][d] * out[i][d];
      }
      float norm = (float) Math.sqrt(sum);
      for (int d = 0; d < dimension; d++) {
        out[i][d] /= norm;
      }
    }
    return out;
  }
}
