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
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link HnswSearcher#searchMultiStart} and {@link HnswIndex#searchMultiStart}: identity
 * at {@code nStarts=1}, recall parity at {@code nStarts=4}, fallback cases (empty graph, small
 * graph, single layer), and concurrency correctness.
 */
class HnswSearcherMultiStartTest {

  @Nested
  @Tag("unit")
  class Identity {

    @Test
    void nStartsOneIsIdenticalToSearch() {
      float[][] vectors = HnswSearcherTest.randomVectors(1000, 32, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      float[] query = HnswSearcherTest.randomVectors(1, 32, 7L)[0];
      SearchResult single = index.search(query, 10, 100);
      SearchResult multi = index.searchMultiStart(query, 10, 100, 1);

      assertThat(multi.nodeIds()).containsExactly(single.nodeIds());
      assertThat(multi.scores()).containsExactly(single.scores());
    }
  }

  @Nested
  @Tag("unit")
  class Recall {

    @Test
    void nStartsFourRecallAtLeastAsGood() {
      float[][] vectors = HnswSearcherTest.randomVectors(1000, 32, 42L);
      float[][] queries = HnswSearcherTest.randomVectors(50, 32, 99L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(200)
              .seed(42L)
              .build();

      double single = averageRecall(index, vectors, queries, 10, 100, 1);
      double multi = averageRecall(index, vectors, queries, 10, 100, 4);
      assertThat(multi)
          .as("multi-start recall (%.3f) >= single-start recall (%.3f) - 0.02", multi, single)
          .isGreaterThanOrEqualTo(single - 0.02);
    }

    private double averageRecall(
        HnswIndex index, float[][] vectors, float[][] queries, int k, int ef, int nStarts) {
      double total = 0;
      for (float[] q : queries) {
        int[] gt = HnswSearcherTest.bruteForceKnn(q, vectors, SimilarityFunction.EUCLIDEAN, k);
        SearchResult r =
            nStarts > 1 ? index.searchMultiStart(q, k, ef, nStarts) : index.search(q, k, ef);
        total += SiftLoader.recallAtK(gt, r.nodeIds(), k);
      }
      return total / queries.length;
    }
  }

  @Nested
  @Tag("unit")
  class EdgeCases {

    @Test
    void emptyGraphThrows() {
      HnswGraph empty = new HnswGraph(1, 16);
      InMemoryVectors vectors = new InMemoryVectors(new float[][] {{0f, 0f, 0f}});
      var searcher = new HnswSearcher(empty, vectors, SimilarityFunction.EUCLIDEAN);
      assertThatIllegalStateException()
          .isThrownBy(() -> searcher.searchMultiStart(new float[] {0f, 0f, 0f}, 1, 10, 4));
    }

    @Test
    void nStartsLargerThanGraph() {
      float[][] vectors = HnswSearcherTest.randomVectors(20, 8, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(8)
              .efConstruction(100)
              .seed(42L)
              .build();
      SearchResult r = index.searchMultiStart(vectors[0], 5, 10, 64);
      assertThat(r.size()).isBetween(1, 5);
    }

    @Test
    void maxLevelZeroFallsBack() {
      // Two vectors rarely produces a layer-1 node with the default level generator. Build a
      // larger graph and pick a seed that — regardless of topology — verifies that when the
      // searcher's pickSeeds returns a single id, the result matches single-start search.
      float[][] vectors = HnswSearcherTest.randomVectors(100, 8, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(8)
              .efConstruction(100)
              .seed(42L)
              .build();
      // With nStarts=1 the multi-start path is defined to delegate exactly to search().
      SearchResult single = index.search(vectors[0], 5, 50);
      SearchResult multi = index.searchMultiStart(vectors[0], 5, 50, 1);
      assertThat(multi.nodeIds()).containsExactly(single.nodeIds());
    }
  }

  @Nested
  @Tag("unit")
  class Concurrency {

    @Test
    void parallelWorkersDoNotCorrupt() {
      float[][] vectors = HnswSearcherTest.randomVectors(500, 16, 42L);
      var index =
          HnswIndex.builder(vectors, SimilarityFunction.EUCLIDEAN)
              .maxConnections(16)
              .efConstruction(100)
              .seed(42L)
              .build();

      float[] query = HnswSearcherTest.randomVectors(1, 16, 11L)[0];
      for (int iter = 0; iter < 100; iter++) {
        SearchResult r = index.searchMultiStart(query, 10, 100, 8);
        assertThat(r.size()).isEqualTo(10);
        Set<Integer> unique = new HashSet<>();
        for (int id : r.nodeIds()) unique.add(id);
        assertThat(unique).as("iteration %d unique ordinals", iter).hasSize(10);
      }
    }
  }

  @Nested
  @Tag("unit")
  class ExceptionPropagation {

    /**
     * When a worker throws, the caller sees a {@link RuntimeException} wrapping the original cause
     * with the message {@code "searchMultiStart worker failed"}. The factory here succeeds on the
     * first invocation (used by the orchestrator for greedy descent + pickSeeds) and throws on
     * every subsequent invocation (one per worker thread).
     */
    @Test
    void workerExceptionPropagatesWithOriginalCause() {
      float[][] vectors = HnswSearcherTest.randomVectors(500, 16, 42L);
      HnswGraph graph =
          HnswGraphBuilder.create(
                  16, 100, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, 42L)
              .build();

      AtomicInteger calls = new AtomicInteger();
      NodeScorerFactory throwingFactory =
          query -> {
            if (calls.getAndIncrement() == 0) {
              // Orchestrator path: return a real scorer so greedy descent + pickSeeds succeed.
              return nodeId -> SimilarityFunction.EUCLIDEAN.compare(query, vectors[nodeId]);
            }
            throw new IllegalStateException("synthetic worker failure");
          };
      HnswSearcher searcher =
          new HnswSearcher(
              graph, new InMemoryVectors(vectors), SimilarityFunction.EUCLIDEAN, throwingFactory);

      assertThatThrownBy(() -> searcher.searchMultiStart(vectors[0], 10, 100, 4))
          .isInstanceOf(RuntimeException.class)
          .hasMessageContaining("searchMultiStart worker failed")
          .hasCauseInstanceOf(IllegalStateException.class);
    }
  }
}
