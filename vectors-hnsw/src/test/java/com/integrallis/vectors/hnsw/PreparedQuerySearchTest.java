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

import static org.junit.jupiter.api.Assertions.*;

import com.integrallis.vectors.core.FusedSimilarity;
import com.integrallis.vectors.core.SimilarityFunction;
import java.lang.foreign.*;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PreparedQuerySearchTest {
  @Test
  void matchesOriginalScorerAcrossFiltersStorageAndMutatedQueries() {
    var random = new SplittableRandom(8273);
    var cosine = SimilarityFunction.COSINE;
    for (int dim : new int[] {7, 65, 128}) {
      float[][] data = new float[192][dim];
      float[] query = new float[dim];
      for (var row : data) for (int d = 0; d < dim; d++) row[d] = (float) random.nextDouble(-1, 1);
      var graph = HnswGraphBuilder.create(8, 64, new InMemoryVectors(data), cosine, 93).build();
      try (var arena = Arena.ofConfined()) {
        var memory = arena.allocate((long) data.length * dim * 4);
        for (int i = 0; i < data.length; i++)
          MemorySegment.copy(data[i], 0, memory, ValueLayout.JAVA_FLOAT, (long) i * dim * 4, dim);
        for (int storage = 0; storage < 3; storage++) {
          boolean segment = storage != 0;
          boolean offsets = storage == 2;
          var offsetReads = new java.util.concurrent.atomic.AtomicInteger();
          RandomAccessVectors source =
              new RandomAccessVectors() {
                public int size() {
                  return data.length;
                }

                public int dimension() {
                  return dim;
                }

                public float[] getVector(int i) {
                  return data[i];
                }

                public boolean sharesReturnBuffer() {
                  return false;
                }

                public boolean supportsSegments() {
                  return segment;
                }

                @Override
                public MemorySegment vectorSegmentStorage() {
                  return offsets ? memory : null;
                }

                @Override
                public long vectorSegmentOffset(int i) {
                  offsetReads.incrementAndGet();
                  return (long) i * dim * 4;
                }

                public MemorySegment vectorSegment(int i) {
                  return memory.asSlice((long) i * dim * 4, (long) dim * 4);
                }
              };
          var actual = new HnswSearcher(graph, source, cosine);
          var expected =
              new HnswSearcher(
                  graph,
                  source,
                  cosine,
                  q -> {
                    float[][] pool = new float[64][];
                    MemorySegment[] rows = new MemorySegment[64];
                    float[] scratch = new float[64];
                    return new NodeScorer() {
                      public float score(int id) {
                        return segment
                            ? cosine.compare(
                                MemorySegment.ofArray(q), source.vectorSegment(id), dim)
                            : cosine.compare(q, data[id]);
                      }

                      public void bulkScore(int[] ids, int offset, int count, float[] out) {
                        if (segment) {
                          for (int i = 0; i < count; i++)
                            rows[i] = source.vectorSegment(ids[offset + i]);
                          FusedSimilarity.bulkCompareSegments(
                              cosine, q, rows, dim, scratch, out, count);
                        } else {
                          for (int i = 0; i < count; i++) pool[i] = data[ids[offset + i]];
                          FusedSimilarity.bulkCompare(cosine, q, pool, scratch, out, count);
                        }
                      }
                    };
                  });
          for (int mutation = 0; mutation < 4; mutation++) {
            for (int d = 0; d < dim; d++) query[d] = (float) random.nextDouble(-1, 1);
            for (int ef : new int[] {10, 32, 128}) {
              same(expected.search(query, 10, ef), actual.search(query, 10, ef));
              same(
                  expected.searchFiltered(query, 10, ef, id -> id % 5 == 0),
                  actual.searchFiltered(query, 10, ef, id -> id % 5 == 0));
            }
          }
          if (offsets && com.integrallis.vectors.core.VectorUtil.supportsCosineNormReuse(dim)) {
            assertTrue(offsetReads.get() > 0);
          } else {
            assertEquals(0, offsetReads.get(), "unaccelerated storage must retain its original scorer");
          }
        }
      }
    }
  }

  private static void same(SearchResult expected, SearchResult actual) {
    assertArrayEquals(expected.nodeIds(), actual.nodeIds());
    // Cold SIMD tier differences are covered by a tight tolerance here. Frozen-JVM benchmarks
    // and the separate kernel probe compare the complete warmed score bits without tolerance.
    assertArrayEquals(expected.scores(), actual.scores(), 1e-6f);
  }
}
