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
class ConcurrentExactScoreReuseTest {
  @Test
  void pruningAvoidsRepeatedVectorReadsDuringConstruction() {
    var vectors = new CountingVectors(792131);
    var graph =
        ConcurrentHnswGraphBuilder.create(16, 200, vectors, SimilarityFunction.COSINE, 42).build(1);
    assertEquals(vectors.size(), graph.size());
    // Main performed 9,010,221 reads on this fixed schedule. Leave room for provider rounding,
    // while requiring removal of at least a million redundant reads, independent of wall time.
    assertTrue(vectors.reads < 8_000_000, "vector reads: " + vectors.reads);
  }

  @Test
  void scoresNeverSurviveIntoAnotherVectorSource() {
    // Repeated ordinal pairs now refer to unrelated vectors. Each build must own its cache.
    var first = new CountingVectors(37);
    var second = new CountingVectors(91);
    var expected =
        ConcurrentHnswGraphBuilder.create(16, 200, second, SimilarityFunction.COSINE, 42).build(1);
    ConcurrentHnswGraphBuilder.create(16, 200, first, SimilarityFunction.COSINE, 42).build(1);
    var actual =
        ConcurrentHnswGraphBuilder.create(16, 200, second, SimilarityFunction.COSINE, 42).build(1);
    for (int n = 0; n < expected.size(); n++) {
      assertEquals(expected.nodeLevel(n), actual.nodeLevel(n));
      for (int level = 0; level <= expected.nodeLevel(n); level++) {
        var a = expected.getNeighbors(n, level);
        var b = actual.getNeighbors(n, level);
        assertEquals(a.size(), b.size());
        for (int i = 0; i < a.size(); i++) {
          assertEquals(a.node(i), b.node(i));
          assertEquals(Float.floatToRawIntBits(a.score(i)), Float.floatToRawIntBits(b.score(i)));
        }
      }
    }
  }

  private static final class CountingVectors implements RandomAccessVectors {
    final float[][] rows = new float[1024][65];
    long reads;

    CountingVectors(long seed) {
      var random = new SplittableRandom(seed);
      for (var row : rows)
        for (int d = 0; d < row.length; d++) row[d] = (float) random.nextDouble(-1, 1);
    }

    public int size() {
      return rows.length;
    }

    public int dimension() {
      return rows[0].length;
    }

    public boolean sharesReturnBuffer() {
      return false;
    }

    public float[] getVector(int n) {
      reads++;
      return rows[n];
    }
  }
}
