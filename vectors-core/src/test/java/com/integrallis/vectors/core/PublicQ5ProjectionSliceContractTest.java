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
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Independent packed Q5_0 encoding and dense algebra for the public unquantized-query API. */
class PublicQ5ProjectionSliceContractTest {
  @ParameterizedTest
  @ValueSource(ints = {32, 64, 96})
  void denseRowsRespectPrefixesAndRejectInvalidSlices(int columns) {
    int rows = 5, rowBytes = columns / 32 * 22;
    byte[] bytes = new byte[rows * rowBytes];
    MemorySegment w = MemorySegment.ofArray(bytes);
    float[] query = new float[columns + 3];
    for (int c = 0; c < columns; c++) query[c] = (c % 11 - 5) * .25f;
    Arrays.fill(query, columns, query.length, Float.NaN);
    float[] expected = new float[rows];
    for (int r = 0; r < rows; r++)
      for (int block = 0; block < columns / 32; block++) {
        int off = r * rowBytes + block * 22;
        w.set(
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
            off,
            (short) 0x3800);
        for (int j = 0; j < 32; j++) {
          int signed = (r * 7 + block * 3 + j * 5) % 32 - 16, encoded = signed + 16;
          bytes[off + 2 + j / 8] |= (byte) ((encoded >>> 4) << (j % 8));
          bytes[off + 6 + j % 16] |= (byte) ((encoded & 15) << (j < 16 ? 0 : 4));
          expected[r] += query[block * 32 + j] * (signed * .5f);
        }
      }
    byte[] original = bytes.clone();
    float[] out = new float[rows + 2];
    Arrays.fill(out, 73f);
    VectorUtil.ggufQ5_0BatchDotProduct(query, w.asReadOnly(), rows, columns, out);
    for (int r = 0; r < rows; r++) {
      assertEquals(expected[r], out[r]);
      assertEquals(
          expected[r],
          VectorUtil.ggufQ5_0DotProduct(
              Arrays.copyOf(query, columns), w.asReadOnly(), r * (long) rowBytes, columns));
    }
    assertEquals(73f, out[rows]);
    assertEquals(73f, out[rows + 1]);
    for (long offset : new long[] {-1, Long.MAX_VALUE, bytes.length - rowBytes + 1L})
      assertThrows(
          IllegalArgumentException.class,
          () -> VectorUtil.ggufQ5_0DotProduct(Arrays.copyOf(query, columns), w, offset, columns));
    for (String fault : new String[] {"query", "output", "weights", "rows", "columns"}) {
      float[] target = new float[fault.equals("output") ? rows - 1 : rows + 2];
      Arrays.fill(target, 73f);
      assertThrows(
          IllegalArgumentException.class,
          () ->
              VectorUtil.ggufQ5_0BatchDotProduct(
                  fault.equals("query") ? new float[columns - 1] : query,
                  fault.equals("weights") ? w.asSlice(0, bytes.length - 1) : w,
                  fault.equals("rows") ? -1 : rows,
                  fault.equals("columns") ? columns - 1 : columns,
                  target));
      for (float value : target) assertEquals(73f, value);
    }
    assertArrayEquals(original, bytes);
  }
}
