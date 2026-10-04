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
package com.integrallis.vectors.core;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PreparedBatchCosineTest {
  @Test
  void preparedBatchesPreserveScoresTailsAndQueryMutation() {
    var random = new SplittableRandom(95231);
    for (int dim : new int[] {1, 7, 16, 31, 32, 65, 100, 128, 129, 512, 768}) {
      float[] query = new float[dim];
      float[][] rows = new float[9][dim];
      MemorySegment[] segments = new MemorySegment[9];
      for (int r = 0; r < rows.length; r++) {
        for (int d = 0; d < dim; d++) rows[r][d] = (float) random.nextDouble(-1, 1);
        segments[r] = MemorySegment.ofArray(rows[r]);
      }
      for (int pass = 0; pass < 2; pass++) {
        for (int d = 0; d < dim; d++) query[d] = (float) random.nextDouble(-1, 1);
        float norm = VectorUtil.batchCosineQueryNorm(query);
        for (int count : new int[] {0, 1, 3, 4, 5, 8, 9}) {
          float[] expected = new float[10], actual = new float[10];
          Arrays.fill(expected, 42f);
          Arrays.fill(actual, 42f);
          VectorUtil.batchCosine(query, rows, expected, count);
          VectorUtil.batchCosineWithQueryNorm(query, rows, norm, actual, count);
          assertArrayEquals(expected, actual, 2e-6f);
          assertEquals(42f, actual[count]);
          VectorUtil.batchCosine(query, segments, dim, expected, count);
          VectorUtil.batchCosineWithQueryNorm(query, segments, dim, norm, actual, count);
          assertArrayEquals(expected, actual, 2e-6f);
          assertEquals(42f, actual[count]);
        }
      }
    }
  }

  @Test
  void validatesArgumentsAndPreservesZeroVectorBehavior() {
    float[] q = new float[32];
    float[][] rows = {q};
    float[] out = new float[1];
    VectorUtil.batchCosineWithQueryNorm(q, rows, 0f, out, 1);
    assertTrue(Float.isNaN(out[0]));
    VectorUtil.batchCosineWithQueryNorm(
        q, new MemorySegment[] {MemorySegment.ofArray(q)}, 32, 0f, out, 1);
    assertTrue(Float.isNaN(out[0]));
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorUtil.batchCosineWithQueryNorm(q, rows, 0f, out, 2));
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorUtil.batchCosineWithQueryNorm(q, rows, 0f, new float[0], 1));
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorUtil.batchCosineWithQueryNorm(q, new MemorySegment[1], 31, 0f, out, 1));
  }
}
