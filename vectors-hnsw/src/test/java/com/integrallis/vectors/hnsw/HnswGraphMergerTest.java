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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class HnswGraphMergerTest {

  @Test
  void emptySuccessorHasNoGraph() {
    HnswGraph old = new HnswGraph(1, 2);
    old.initNode(0, 0);
    old.setEntryNode(0, 0);

    HnswGraph merged =
        HnswGraphMerger.merge(old, new float[0][], new int[] {-1}, SimilarityFunction.COSINE, 2, 8);

    assertThat(merged).isNull();
  }

  @Test
  void mergeRemapsSurvivorsRepairsConnectivityAndPreservesLevels() {
    int oldSize = 40;
    int dimension = 8;
    int maxConnections = 4;
    float[][] oldVectors = randomVectors(oldSize, dimension, 42L);
    HnswGraph old =
        HnswGraphBuilder.create(
                maxConnections,
                40,
                new InMemoryVectors(oldVectors),
                SimilarityFunction.EUCLIDEAN,
                42L)
            .build();

    int[] oldToNew = new int[oldSize];
    Arrays.fill(oldToNew, -1);
    List<float[]> survivors = new ArrayList<>();
    List<Integer> survivingOldOrdinals = new ArrayList<>();
    for (int oldOrdinal = 0; oldOrdinal < oldSize; oldOrdinal++) {
      if (oldOrdinal % 4 != 0) {
        oldToNew[oldOrdinal] = survivors.size();
        survivors.add(oldVectors[oldOrdinal]);
        survivingOldOrdinals.add(oldOrdinal);
      }
    }
    float[][] newVectors = survivors.toArray(float[][]::new);

    HnswGraph merged =
        HnswGraphMerger.merge(
            old, newVectors, oldToNew, SimilarityFunction.EUCLIDEAN, maxConnections, 40);

    assertThat(merged).isNotNull();
    assertThat(merged.size()).isEqualTo(newVectors.length);
    assertThat(merged.maxConnections()).isEqualTo(maxConnections);

    int highestLevel = -1;
    for (int newOrdinal = 0; newOrdinal < merged.size(); newOrdinal++) {
      int oldOrdinal = survivingOldOrdinals.get(newOrdinal);
      assertThat(merged.nodeLevel(newOrdinal)).isEqualTo(old.nodeLevel(oldOrdinal));
      highestLevel = Math.max(highestLevel, merged.nodeLevel(newOrdinal));

      NeighborArray neighbors = merged.getNeighbors(newOrdinal, 0);
      assertThat(neighbors.size()).isLessThanOrEqualTo(2 * maxConnections);
      for (int i = 0; i < neighbors.size(); i++) {
        assertThat(neighbors.node(i)).isBetween(0, merged.size() - 1);
        assertThat(neighbors.node(i)).isNotEqualTo(newOrdinal);
      }
    }
    assertThat(merged.maxLevel()).isEqualTo(highestLevel);
    assertThat(merged.nodeLevel(merged.entryNode())).isEqualTo(highestLevel);

    HnswSearcher searcher =
        new HnswSearcher(merged, new InMemoryVectors(newVectors), SimilarityFunction.EUCLIDEAN);
    for (int ordinal = 0; ordinal < newVectors.length; ordinal++) {
      SearchResult result = searcher.search(newVectors[ordinal], 1, 40);
      assertThat(result.nodeIds()).containsExactly(ordinal);
    }
  }

  private static float[][] randomVectors(int count, int dimension, long seed) {
    Random random = new Random(seed);
    float[][] vectors = new float[count][dimension];
    for (float[] vector : vectors) {
      for (int d = 0; d < dimension; d++) {
        vector[d] = random.nextFloat() * 2f - 1f;
      }
    }
    return vectors;
  }
}
