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
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Regression guard: the multi-threaded HNSW builder must never emit a self-loop (a node listed as
 * its own neighbour). The race — a concurrent insert back-links the in-flight node into the graph,
 * and its maximal self-similarity then pulls it into its own neighbour selection — only manifests
 * with enough nodes and threads, so this builds several graphs across seeds at high parallelism and
 * scans every edge. Before the fix it threw "layer 0 self-loop on node N" on commit (via
 * HnswGraphCodec); here we assert directly on the graph so the failure is local.
 */
@Tag("unit")
class ConcurrentHnswGraphBuilderSelfLoopTest {

  private static final int N = 2_000;
  private static final int DIM = 12;

  private static float[][] randomVectors(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] v = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) {
        v[i][d] = (float) rng.nextGaussian();
      }
    }
    return v;
  }

  @Test
  void parallelBuildHasNoSelfLoopsOrOutOfRangeOrDuplicateNeighbors() {
    int parallelism = Math.max(4, Runtime.getRuntime().availableProcessors());
    for (long seed = 0; seed < 6; seed++) {
      var rav = new InMemoryVectors(randomVectors(N, DIM, seed));
      HnswGraph graph =
          ConcurrentHnswGraphBuilder.create(16, 200, rav, SimilarityFunction.EUCLIDEAN, seed)
              .build(parallelism);

      assertThat(graph.size()).isEqualTo(N);
      for (int node = 0; node < N; node++) {
        int levels = graph.nodeLevel(node) + 1;
        for (int layer = 0; layer < levels; layer++) {
          NeighborArray na = graph.getNeighbors(node, layer);
          if (na == null) {
            continue;
          }
          java.util.BitSet seen = new java.util.BitSet(N);
          for (int k = 0; k < na.size(); k++) {
            int nbr = na.node(k);
            assertThat(nbr)
                .as("seed %d node %d layer %d neighbour %d in range", seed, node, layer, k)
                .isBetween(0, N - 1);
            assertThat(nbr)
                .as("seed %d: self-loop at node %d layer %d", seed, node, layer)
                .isNotEqualTo(node);
            assertThat(seen.get(nbr))
                .as("seed %d: duplicate neighbour %d at node %d layer %d", seed, nbr, node, layer)
                .isFalse();
            seen.set(nbr);
          }
        }
      }
    }
  }
}
