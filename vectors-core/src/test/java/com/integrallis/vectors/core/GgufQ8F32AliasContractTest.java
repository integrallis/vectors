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

class GgufQ8F32AliasContractTest {
  private static GgufQ8_0F32Matrix matrix() {
    byte[] bytes = new byte[68];
    MemorySegment w = MemorySegment.ofArray(bytes);
    for (int row = 0; row < 2; row++) {
      w.set(
          ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
          row * 34L,
          (short) 0x3c00);
      Arrays.fill(bytes, row * 34 + 2, (row + 1) * 34, (byte) 1);
    }
    return GgufQ8_0F32Matrix.from(w, 2, 32);
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 4, 5})
  void overlapRejectsBeforeMutationInBothDirections(int batches) {
    GgufQ8_0F32Matrix m = matrix();
    for (int[] offsets :
        new int[][] {{0, 0}, {0, 1}, {1, 0}, {10, 9}, {10, 10 + 32 * batches - 1}}) {
      float[] values = new float[64 * batches + 40];
      Arrays.fill(values, 1f);
      float[] before = values.clone();
      assertThrows(
          IllegalArgumentException.class,
          () -> m.multiplyBatch(values, offsets[0], batches, values, offsets[1]));
      assertArrayEquals(before, values);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 4, 5})
  void disjointRangesInSameArrayAreSupported(int batches) {
    GgufQ8_0F32Matrix m = matrix();
    for (boolean outputFirst : new boolean[] {false, true}) {
      int in = outputFirst ? 2 * batches : 0, out = outputFirst ? 0 : 32 * batches;
      float[] values = new float[34 * batches + 3];
      Arrays.fill(values, 73f);
      for (int b = 0; b < batches; b++) Arrays.fill(values, in + b * 32, in + (b + 1) * 32, b + 1);
      float[] before = values.clone();
      m.multiplyBatch(values, in, batches, values, out);
      for (int i = 0; i < values.length; i++) {
        if (i >= out && i < out + 2 * batches) assertEquals(32f * ((i - out) / 2 + 1), values[i]);
        else assertEquals(before[i], values[i]);
      }
    }
  }
}
