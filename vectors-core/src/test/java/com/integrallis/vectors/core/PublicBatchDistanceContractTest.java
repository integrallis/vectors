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
import java.util.Arrays;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Dense algebra, row-prefix ownership and read-only segment contracts for batched distances. */
class PublicBatchDistanceContractTest {
  @ParameterizedTest
  @ValueSource(ints = {1, 3, 7, 9, 17, 37, 257})
  void fourRowGroupsAndRemaindersMatchDenseAlgebra(int dimensions) {
    for (int count : new int[] {4, 5, 8, 9}) {
      float[] query = new float[dimensions];
      float[][] rows = new float[count + 1][dimensions];
      for (int c = 0; c < dimensions; c++) {
        query[c] = (c % 7 + 1) * .25f;
        for (int r = 0; r < count; r++) rows[r][c] = ((c * 3 + r * 5) % 13 - 6) * .5f;
      }
      float[] originalQuery = query.clone();
      float[][] originals = Arrays.stream(rows).map(float[]::clone).toArray(float[][]::new);
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment[] segments = new MemorySegment[count + 1];
        for (int r = 0; r < count; r++) {
          MemorySegment segment = arena.allocate(dimensions * 4L, 4);
          segment.copyFrom(MemorySegment.ofArray(rows[r]));
          segments[r] = segment.asReadOnly();
        }
        // Unused row is deliberately null: the public count is the ownership boundary.
        rows[count] = null;
        for (int operation = 0; operation < 3; operation++) {
          float[] arrayOut = new float[count + 2], segmentOut = new float[count + 2];
          Arrays.fill(arrayOut, 123456f);
          Arrays.fill(segmentOut, 123456f);
          if (operation == 0) {
            VectorUtil.batchDotProduct(query, rows, arrayOut, count);
            VectorUtil.batchDotProduct(query, segments, dimensions, segmentOut, count);
          } else if (operation == 1) {
            VectorUtil.batchSquaredL2(query, rows, arrayOut, count);
            VectorUtil.batchSquaredL2(query, segments, dimensions, segmentOut, count);
          } else {
            VectorUtil.batchCosine(query, rows, arrayOut, count);
            VectorUtil.batchCosine(query, segments, dimensions, segmentOut, count);
          }
          for (int r = 0; r < count; r++) {
            double dot = 0, qn = 0, rn = 0, distance = 0;
            for (int c = 0; c < dimensions; c++) {
              double q = query[c], v = rows[r][c], d = q - v;
              dot += q * v;
              qn += q * q;
              rn += v * v;
              distance += d * d;
            }
            float expected =
                (float)
                    (operation == 0 ? dot : operation == 1 ? distance : dot / Math.sqrt(qn * rn));
            assertEquals(expected, arrayOut[r], operation == 2 ? 1e-6f : 0f);
            assertEquals(expected, segmentOut[r], operation == 2 ? 1e-6f : 0f);
          }
          for (int i = count; i < arrayOut.length; i++) {
            assertEquals(123456f, arrayOut[i]);
            assertEquals(123456f, segmentOut[i]);
          }
        }
        assertArrayEquals(originalQuery, query);
        for (int r = 0; r < count; r++) {
          assertArrayEquals(originals[r], rows[r]);
          assertArrayEquals(originals[r], segments[r].toArray(ValueLayout.JAVA_FLOAT));
        }
      }
    }
  }

  @org.junit.jupiter.api.Test
  void negativeGeometryAndNullActiveRowsRejectBeforeWrites() {
    float[] query = {1, 2, 3}, matrix = {1, 2, 3, 4, 5, 6}, out = {73, 73, 73};
    float[] valid = new float[2];
    VectorUtil.batchDotProduct(query, matrix, 2, 3, valid);
    assertArrayEquals(new float[] {14, 32}, valid);
    VectorUtil.batchDotProductExact(query, 0, matrix, 0, 3, 2, 3, valid, 0);
    assertArrayEquals(new float[] {14, 32}, valid);
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorUtil.batchDotProduct(query, matrix, -1, 3, out));
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorUtil.batchDotProduct(query, matrix, 2, -1, out));
    for (int offsetIndex = 0; offsetIndex < 3; offsetIndex++) {
      int q = offsetIndex == 0 ? -1 : 0,
          m = offsetIndex == 1 ? -1 : 0,
          o = offsetIndex == 2 ? -1 : 0;
      assertThrows(
          IllegalArgumentException.class,
          () -> VectorUtil.batchDotProductExact(query, q, matrix, m, 3, 2, 3, out, o));
    }
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorUtil.batchDotProductExact(query, 0, matrix, 0, 3, -1, 3, out, 0));
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorUtil.batchDotProductExact(query, 0, matrix, 0, 3, 2, -1, out, 0));
    // Empty row range is valid even at the end of both caller-owned storage ranges.
    VectorUtil.batchDotProductExact(query, 0, matrix, matrix.length, 3, 0, 3, out, out.length);
    assertThrows(
        NullPointerException.class,
        () -> VectorUtil.batchDotProduct(query, new float[][] {query, null}, out, 2));
    assertThrows(
        NullPointerException.class,
        () ->
            VectorUtil.batchDotProduct(
                query, new MemorySegment[] {MemorySegment.ofArray(query), null}, 3, out, 2));
    assertArrayEquals(new float[] {73, 73, 73}, out);
  }
}
