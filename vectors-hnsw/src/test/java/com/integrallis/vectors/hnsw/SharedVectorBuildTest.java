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
class SharedVectorBuildTest {
  @Test
  void reusedReadBufferPreservesSerialGraph() {
    float[][] rows = rows();
    HnswGraph expected = build(source(rows, false));
    HnswGraph actual = build(source(rows, true));
    assertGraph(expected, actual);
  }

  private static HnswGraph build(RandomAccessVectors source) {
    return HnswGraphBuilder.create(16, 200, source, SimilarityFunction.EUCLIDEAN, 42).build();
  }

  private static RandomAccessVectors source(float[][] rows, boolean reuse) {
    return new RandomAccessVectors() {
      private final float[] buffer = new float[16];

      public int size() {
        return rows.length;
      }

      public int dimension() {
        return 16;
      }

      // Both fixtures retain the interface's conservative sharesReturnBuffer=true default,
      // so both take the same scoring path. Only the actual aliasing differs.
      public float[] getVector(int id) {
        if (!reuse) return rows[id];
        System.arraycopy(rows[id], 0, buffer, 0, buffer.length);
        return buffer;
      }
    };
  }

  private static float[][] rows() {
    float[][] rows = new float[300][16];
    SplittableRandom random = new SplittableRandom(9741);
    for (float[] row : rows)
      for (int d = 0; d < row.length; d++) row[d] = (float) random.nextDouble();
    return rows;
  }

  private static void assertGraph(HnswGraph expected, HnswGraph actual) {
    assertEquals(expected.size(), actual.size());
    assertEquals(expected.entryNode(), actual.entryNode());
    for (int i = 0; i < expected.size(); i++) {
      assertEquals(expected.nodeLevel(i), actual.nodeLevel(i));
      for (int layer = 0; layer <= expected.nodeLevel(i); layer++) {
        NeighborArray a = expected.getNeighbors(i, layer), b = actual.getNeighbors(i, layer);
        assertEquals(a.size(), b.size());
        for (int n = 0; n < a.size(); n++) {
          assertEquals(a.node(n), b.node(n));
          assertEquals(Float.floatToRawIntBits(a.score(n)), Float.floatToRawIntBits(b.score(n)));
        }
      }
    }
  }
}
