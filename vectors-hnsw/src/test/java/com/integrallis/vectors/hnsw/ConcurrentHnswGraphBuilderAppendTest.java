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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/**
 * {@link ConcurrentHnswGraphBuilder#append} has to produce a graph that searches as well as a full
 * rebuild, or it is not a substitute for one.
 */
class ConcurrentHnswGraphBuilderAppendTest {

  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;
  private static final int DIM = 64;
  private static final int M = 16;
  private static final int EF = 100;
  private static final long SEED = 7L;

  @Test
  void appendedGraphRetrievesAsWellAsARebuiltOne() {
    float[][] vectors = randomUnitVectors(4_000, DIM, SEED);
    int carried = 2_500;

    HnswGraph rebuilt = build(vectors, vectors.length);
    HnswGraph appended = buildThenAppend(vectors, carried);

    assertThat(appended.size()).isEqualTo(vectors.length);

    float[][] queries = randomUnitVectors(200, DIM, SEED + 1);
    double rebuiltRecall = recallAt10(rebuilt, vectors, queries);
    double appendedRecall = recallAt10(appended, vectors, queries);

    // The appended graph is a different graph, not a reproduction of the rebuilt one, so this is a
    // quality floor rather than an equality check.
    assertThat(appendedRecall)
        .as("recall@10 after appending %d to %d", vectors.length - carried, carried)
        .isGreaterThan(0.90)
        .isGreaterThan(rebuiltRecall - 0.05);
  }

  @Test
  void appendedNodesAreReachableFromTheGraph() {
    float[][] vectors = randomUnitVectors(1_500, DIM, SEED);
    int carried = 1_000;
    HnswGraph appended = buildThenAppend(vectors, carried);

    HnswSearcher searcher = new HnswSearcher(appended, new InMemoryVectors(vectors), METRIC);
    int found = 0;
    for (int id = carried; id < vectors.length; id++) {
      SearchResult result = searcher.search(vectors[id], 1, EF);
      if (result.nodeIds().length > 0 && result.nodeIds()[0] == id) {
        found++;
      }
    }
    int appendedCount = vectors.length - carried;
    assertThat(found)
        .as("appended nodes found as their own nearest neighbour")
        .isGreaterThan((int) (appendedCount * 0.98));
  }

  @Test
  void everyLayerZeroArrayRespectsTheConnectionLimit() {
    float[][] vectors = randomUnitVectors(1_200, DIM, SEED);
    HnswGraph appended = buildThenAppend(vectors, 800);
    int limit = appended.maxConnections0();
    for (int node = 0; node < appended.size(); node++) {
      NeighborArray neighbours = appended.getNeighbors(node, 0);
      assertThat(neighbours.size()).as("node %d layer-0 degree", node).isLessThanOrEqualTo(limit);
    }
  }

  @Test
  void carriedOverEdgesSurvive() {
    float[][] vectors = randomUnitVectors(900, DIM, SEED);
    int carried = 600;
    HnswGraph old = build(Arrays.copyOf(vectors, carried), carried);
    HnswGraph appended =
        ConcurrentHnswGraphBuilder.create(M, EF, new InMemoryVectors(vectors), METRIC, SEED)
            .append(old, carried, 2);

    for (int node = 0; node < carried; node++) {
      assertThat(appended.nodeLevel(node)).isEqualTo(old.nodeLevel(node));
      int[] before = ids(old.getNeighbors(node, 0));
      int[] after = ids(appended.getNeighbors(node, 0));
      // Appending may add backlinks and prune for diversity, but the node must keep neighbours.
      assertThat(after.length).as("node %d kept neighbours", node).isGreaterThan(0);
      assertThat(before.length).isGreaterThan(0);
    }
  }

  @Test
  void appendingNothingIsTheSameGraph() {
    float[][] vectors = randomUnitVectors(500, DIM, SEED);
    HnswGraph old = build(vectors, vectors.length);
    HnswGraph appended =
        ConcurrentHnswGraphBuilder.create(M, EF, new InMemoryVectors(vectors), METRIC, SEED)
            .append(old, vectors.length, 2);
    assertThat(appended.size()).isEqualTo(old.size());
    for (int node = 0; node < old.size(); node++) {
      assertThat(ids(appended.getNeighbors(node, 0))).isEqualTo(ids(old.getNeighbors(node, 0)));
    }
  }

  @Test
  void mismatchedInputsAreRejected() {
    float[][] vectors = randomUnitVectors(300, DIM, SEED);
    HnswGraph old = build(Arrays.copyOf(vectors, 200), 200);
    var builder =
        ConcurrentHnswGraphBuilder.create(M, EF, new InMemoryVectors(vectors), METRIC, SEED);

    assertThatThrownBy(() -> builder.append(null, 200, 1)).isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> builder.append(old, 150, 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("holds 200 nodes but firstNewOrdinal is 150");
    assertThatThrownBy(() -> builder.append(old, 400, 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("firstNewOrdinal must be in [0, 300]");
    assertThatThrownBy(
            () ->
                ConcurrentHnswGraphBuilder.create(
                        M * 2, EF, new InMemoryVectors(vectors), METRIC, SEED)
                    .append(old, 200, 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("M=");
  }

  // -------------------------------------------------------------------------

  private static HnswGraph build(float[][] vectors, int count) {
    float[][] slice = vectors.length == count ? vectors : Arrays.copyOf(vectors, count);
    return ConcurrentHnswGraphBuilder.create(M, EF, new InMemoryVectors(slice), METRIC, SEED)
        .build(4);
  }

  private static HnswGraph buildThenAppend(float[][] vectors, int carried) {
    HnswGraph old = build(vectors, carried);
    return ConcurrentHnswGraphBuilder.create(M, EF, new InMemoryVectors(vectors), METRIC, SEED)
        .append(old, carried, 4);
  }

  private static double recallAt10(HnswGraph graph, float[][] vectors, float[][] queries) {
    HnswSearcher searcher = new HnswSearcher(graph, new InMemoryVectors(vectors), METRIC);
    int hits = 0;
    for (float[] query : queries) {
      List<Integer> truth = bruteForceTop(query, vectors, 10);
      SearchResult result = searcher.search(query, 10, EF);
      for (int id : result.nodeIds()) {
        if (truth.contains(id)) {
          hits++;
        }
      }
    }
    return hits / (double) (queries.length * 10);
  }

  private static List<Integer> bruteForceTop(float[] query, float[][] vectors, int k) {
    Integer[] order = new Integer[vectors.length];
    for (int i = 0; i < vectors.length; i++) {
      order[i] = i;
    }
    Arrays.sort(
        order,
        (a, b) ->
            Float.compare(METRIC.compare(query, vectors[b]), METRIC.compare(query, vectors[a])));
    List<Integer> top = new ArrayList<>(k);
    for (int i = 0; i < k; i++) {
      top.add(order[i]);
    }
    return top;
  }

  private static int[] ids(NeighborArray array) {
    int[] out = new int[array.size()];
    for (int i = 0; i < array.size(); i++) {
      out[i] = array.node(i);
    }
    Arrays.sort(out);
    return out;
  }

  private static float[][] randomUnitVectors(int count, int dim, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    float[][] out = new float[count][dim];
    for (int i = 0; i < count; i++) {
      double sum = 0;
      for (int d = 0; d < dim; d++) {
        out[i][d] = (float) random.nextGaussian();
        sum += (double) out[i][d] * out[i][d];
      }
      float norm = (float) Math.sqrt(sum);
      for (int d = 0; d < dim; d++) {
        out[i][d] /= norm;
      }
    }
    return out;
  }
}
