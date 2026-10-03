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

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ExactScoreCacheTest {
  @Test
  void repeatPruningReusesExactScoresWithoutChangingSelectedEdges() {
    for (int storage = 0; storage < 3; storage++)
      for (SimilarityFunction metric : SimilarityFunction.values()) {
        CountingVectors vectors = new CountingVectors(storage);
        NeighborArray candidates = new NeighborArray(65);
        for (int i = 1; i < 65; i++)
          candidates.insert(i, metric.compare(vectors.rows[0], vectors.rows[i]));
        NeighborArray expected = NeighborSelector.selectDiverse(candidates, 16, vectors, metric);
        ExactScoreCache cache = new ExactScoreCache();
        vectors.reads = 0;
        NeighborArray cold = NeighborSelector.selectDiverse(candidates, 16, vectors, metric, cache);
        int coldReads = vectors.reads;
        NeighborArray warm = NeighborSelector.selectDiverse(candidates, 16, vectors, metric, cache);
        int warmReads = vectors.reads - coldReads;
        assertTrue(
            warmReads < coldReads / 2, "repeated pruning should avoid repeated vector reads");
        assertSameEdges(expected, cold);
        assertSameEdges(expected, warm);
      }
  }

  @Test
  void collisionsNeverReturnAnotherPairsScore() {
    ExactScoreCache cache = new ExactScoreCache();
    for (int i = 0; i < 100000; i++) {
      long key = ExactScoreCache.key(i, i + 1);
      cache.put(key, i / 100000f);
      assertEquals(Float.floatToRawIntBits(i / 100000f), Float.floatToRawIntBits(cache.get(key)));
      assertEquals(key, ExactScoreCache.key(i + 1, i));
    }
    // Old entries may have been evicted, but a surviving entry must retain its exact value.
    for (int i = 0; i < 100000; i++) {
      float score = cache.get(ExactScoreCache.key(i, i + 1));
      if (score >= 0f)
        assertEquals(Float.floatToRawIntBits(i / 100000f), Float.floatToRawIntBits(score));
    }
  }

  private static void assertSameEdges(NeighborArray expected, NeighborArray actual) {
    assertEquals(expected.size(), actual.size());
    for (int i = 0; i < expected.size(); i++) {
      assertEquals(expected.node(i), actual.node(i));
      assertEquals(
          Float.floatToRawIntBits(expected.score(i)), Float.floatToRawIntBits(actual.score(i)));
    }
  }

  private static class CountingVectors implements RandomAccessVectors {
    final float[][] rows = new float[65][32];
    int reads;
    final int storage;
    final float[] shared = new float[32];

    CountingVectors(int storage) {
      this.storage = storage;
      SplittableRandom random = new SplittableRandom(3752);
      for (float[] row : rows)
        for (int d = 0; d < row.length; d++) row[d] = (float) random.nextDouble() / 8;
    }

    public int size() {
      return rows.length;
    }

    public int dimension() {
      return 32;
    }

    public boolean sharesReturnBuffer() {
      return storage == 1;
    }

    public boolean supportsSegments() {
      return storage == 2;
    }

    public java.lang.foreign.MemorySegment vectorSegment(int id) {
      reads++;
      return java.lang.foreign.MemorySegment.ofArray(rows[id]);
    }

    public float[] getVector(int id) {
      reads++;
      if (storage != 1) return rows[id];
      System.arraycopy(rows[id], 0, shared, 0, shared.length);
      return shared;
    }
  }
}
