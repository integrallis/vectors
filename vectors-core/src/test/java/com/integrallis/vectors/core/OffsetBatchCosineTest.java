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

import java.lang.foreign.*;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class OffsetBatchCosineTest {
  @Test
  void validatesBoundsAndKeepsScalarAndZeroVectorSemantics() {
    float[] query = {1f, 2f, -3f};
    var matrix = MemorySegment.ofArray(new float[] {2f, -1f, 4f, 3f, 2f, 1f});
    long[] offsets = {12, 0, 12};
    MemorySegment[] rows = {matrix.asSlice(12), matrix.asSlice(0, 12), matrix.asSlice(12)};
    var scalar = new ScalarVectorUtilSupport();
    float[] expected = new float[3], actual = new float[3];
    scalar.batchCosine(query, rows, 3, expected, 3);
    scalar.batchCosineWithQueryNorm(query, matrix, offsets, 3,
        scalar.batchCosineQueryNorm(query), actual, 3);
    assertArrayEquals(expected, actual);
    for (int count : new int[] {-1, 4}) {
      assertThrows(IllegalArgumentException.class, () ->
          VectorUtil.batchCosineWithQueryNorm(query, matrix, offsets, 3, 14f, actual, count));
    }
    assertThrows(IllegalArgumentException.class, () ->
        VectorUtil.batchCosineWithQueryNorm(query, matrix, offsets, 2, 14f, actual, 3));
    assertThrows(IllegalArgumentException.class, () ->
        VectorUtil.batchCosineWithQueryNorm(query, matrix, offsets, 3, 14f, new float[1], 3));
    assertThrows(IndexOutOfBoundsException.class, () ->
        VectorUtil.batchCosineWithQueryNorm(query, matrix, new long[] {16}, 3, 14f, actual, 1));
    float[] zero = new float[3];
    VectorUtil.batchCosineWithQueryNorm(zero, matrix, offsets, 3, 0f, actual, 3);
    for (float score : actual) assertTrue(Float.isNaN(score));
    // Empty batches do not touch unselected offsets.
    VectorUtil.batchCosineWithQueryNorm(query, matrix, new long[] {-1}, 3, 14f, actual, 0);
  }

  @Test
  void offsetsPreserveRowOrderPaddingDuplicatesAndTails() {
    var random = new SplittableRandom(7353);
    try (var arena = Arena.ofConfined()) {
      for (int dim : new int[] {1, 7, 16, 31, 32, 65, 100, 129, 512, 768}) {
        long stride = (long) (dim + 13) * 4;
        var matrix = arena.allocate(stride * 9 + 64);
        float[] query = new float[dim];
        for (int d = 0; d < dim; d++) query[d] = (float) random.nextDouble(-1, 1);
        long[] offsets = new long[9];
        MemorySegment[] rows = new MemorySegment[9];
        for (int r = 0; r < 9; r++) {
          offsets[r] = 64 + (r * 5 % 9) * stride;
          rows[r] = matrix.asSlice(offsets[r], (long) dim * 4);
          for (int d = 0; d < dim; d++)
            rows[r].setAtIndex(ValueLayout.JAVA_FLOAT, d, (float) random.nextDouble(-1, 1));
        }
        offsets[7] = offsets[2];
        rows[7] = rows[2];
        for (int count = 0; count <= 9; count++) {
          float[] expected = new float[10], actual = new float[10];
          expected[9] = actual[9] = 42f;
          VectorUtil.batchCosine(query, rows, dim, expected, count);
          VectorUtil.batchCosineWithQueryNorm(
              query, matrix, offsets, dim, VectorUtil.batchCosineQueryNorm(query), actual, count);
          assertArrayEquals(expected, actual, 2e-6f);
          assertEquals(42f, actual[9]);
        }
        float[] actual = new float[1];
        assertThrows(
            IndexOutOfBoundsException.class,
            () ->
                VectorUtil.batchCosineWithQueryNorm(
                    query, matrix, new long[] {-4}, dim, 1f, actual, 1));
        assertThrows(
            IndexOutOfBoundsException.class,
            () ->
                VectorUtil.batchCosineWithQueryNorm(
                    query, matrix, new long[] {Long.MAX_VALUE}, dim, 1f, actual, 1));
      }
    }
  }
}
