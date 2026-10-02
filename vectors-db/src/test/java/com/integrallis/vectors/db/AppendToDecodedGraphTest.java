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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.storage.HnswGraphCodec;
import com.integrallis.vectors.hnsw.ConcurrentHnswGraphBuilder;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.hnsw.HnswIndex;
import com.integrallis.vectors.hnsw.InMemoryVectors;
import com.integrallis.vectors.hnsw.SearchResult;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/**
 * {@link HnswGraphCodec} stores node ids without scores, so a decoded graph's neighbour arrays
 * carry synthetic monotonically-decreasing scores. That is sound while a committed graph is
 * read-only, and wrong the moment one is extended: insertion and diversity pruning order neighbours
 * by score, so carrying synthetic values into an append degrades the graph instead of failing.
 *
 * <p>This is the regression guard for that. A persistent collection always appends onto a decoded
 * graph, so an append that only works on an in-heap graph is no fix at all.
 */
class AppendToDecodedGraphTest {

  private static final SimilarityFunction METRIC = SimilarityFunction.COSINE;
  private static final int M = 16;
  private static final int EF = 100;
  private static final int CARRIED = 500;
  private static final int TOTAL = 1_000;

  @Test
  void appendingToADecodedGraphIsAsGoodAsAppendingToTheOriginal() throws Exception {
    float[][] vectors = randomUnitVectors(TOTAL, 32, 5L);
    HnswGraph inHeap =
        ConcurrentHnswGraphBuilder.create(
                M, EF, new InMemoryVectors(Arrays.copyOf(vectors, CARRIED)), METRIC, 1L)
            .build(2);
    HnswGraph decoded = HnswGraphCodec.decode(HnswGraphCodec.encode(inHeap));

    assertThat(decoded.size()).isEqualTo(inHeap.size());
    assertThat(decoded.entryNode()).isEqualTo(inHeap.entryNode());
    assertThat(decoded.maxLevel()).isEqualTo(inHeap.maxLevel());

    int fromInHeap = selfRetrieved(append(inHeap, vectors), vectors);
    int fromDecoded = selfRetrieved(append(decoded, vectors), vectors);

    assertThat(fromInHeap).as("appended onto the in-heap graph").isGreaterThan(TOTAL - 10);
    assertThat(fromDecoded)
        .as("appended onto a graph decoded from graph.bin")
        .isGreaterThan(TOTAL - 10);
  }

  private static HnswGraph append(HnswGraph old, float[][] vectors) {
    return ConcurrentHnswGraphBuilder.create(M, EF, new InMemoryVectors(vectors), METRIC, 2L)
        .append(old, CARRIED, 2);
  }

  /** How many vectors are their own nearest neighbour — the cheapest proof a graph is navigable. */
  private static int selfRetrieved(HnswGraph graph, float[][] vectors) {
    HnswIndex index = HnswIndex.ofPrebuilt(graph, new InMemoryVectors(vectors), METRIC);
    int found = 0;
    for (int i = 0; i < vectors.length; i++) {
      SearchResult result = index.search(vectors[i], 1, EF);
      if (result.nodeIds().length > 0 && result.nodeIds()[0] == i) {
        found++;
      }
    }
    return found;
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
