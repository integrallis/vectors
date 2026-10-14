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
import java.util.*;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.*;

/** Caller-owned prequantized row slices must validate before writing any output. */
class PublicPrequantizedRowRangeContractTest {
  static Stream<Arguments> cases() {
    Stream.Builder<Arguments> s = Stream.builder();
    for (String format : new String[] {"q4", "q8", "block-q8"})
      for (String fault :
          new String[] {
            "valid",
            "batch",
            "rows",
            "from",
            "empty",
            "end",
            "capacity",
            "dimensions",
            "output",
            "weights",
            "null-output",
            "null-activation"
          }) s.add(Arguments.of(format, fault));
    return s.build();
  }

  @ParameterizedTest(name = "{0} {1}")
  @MethodSource("cases")
  void validatesRangeAndPreservesUnownedRows(String format, String fault) {
    int blockBytes = format.equals("q4") ? 18 : 34;
    byte[] bytes = new byte[4 * 2 * blockBytes];
    MemorySegment weights = MemorySegment.ofArray(bytes);
    for (int block = 0; block < 8; block++) {
      weights.set(
          ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
          block * (long) blockBytes,
          (short) 0x3c00);
      Arrays.fill(
          bytes,
          block * blockBytes + 2,
          (block + 1) * blockBytes,
          (byte) (format.equals("q4") ? 0x99 : 1));
    }
    GgufQ8ActivationLayout layout =
        format.equals("block-q8")
            ? GgufQ8ActivationLayout.BLOCK_MAJOR_BYTES
            : GgufQ8ActivationLayout.PACKED_BYTES;
    GgufQ8_0Batch activation = GgufQ8_0Batch.allocate(2, 32, layout);
    float[] input = new float[64];
    Arrays.fill(input, 127f);
    activation.quantizeForQ4(input, 2, GgufQ4Kernel.WIDENED);
    float[] control = new float[10];
    Arrays.fill(control, 73f);
    call(format, weights, 2, 4, 32, 1, 3, control, activation);
    assertArrayEquals(new float[] {73, 4064, 4064, 73, 73, 4064, 4064, 73, 73, 73}, control);
    if (fault.equals("valid")) return;
    float[] out = new float[fault.equals("output") ? 7 : 10];
    Arrays.fill(out, 73f);
    int batch = fault.equals("batch") ? 0 : fault.equals("capacity") ? 3 : 2;
    int rows = fault.equals("rows") ? 0 : 4;
    int columns = fault.equals("dimensions") ? 64 : 32;
    int from = fault.equals("from") ? -1 : 1;
    int to = fault.equals("empty") ? 1 : fault.equals("end") ? 5 : 3;
    MemorySegment w = fault.equals("weights") ? weights.asSlice(0, 1) : weights;
    Class<? extends Throwable> type =
        fault.startsWith("null-")
            ? NullPointerException.class
            : Set.of("from", "empty", "end").contains(fault)
                ? IndexOutOfBoundsException.class
                : IllegalArgumentException.class;
    assertThrows(
        type,
        () ->
            call(
                format,
                w,
                batch,
                rows,
                columns,
                from,
                to,
                fault.equals("null-output") ? null : out,
                fault.equals("null-activation") ? null : activation));
    for (float v : out) assertEquals(73f, v);
    assertArrayEquals(bytes, weights.toArray(ValueLayout.JAVA_BYTE));
  }

  private static void call(
      String format,
      MemorySegment w,
      int b,
      int r,
      int c,
      int from,
      int to,
      float[] out,
      GgufQ8_0Batch a) {
    switch (format) {
      case "q4" ->
          VectorUtil.ggufQ4_0Q8_0BatchedMatmulRows(
              w, b, r, c, from, to, out, a, new float[2 * 4 * 8], GgufQ4Kernel.WIDENED);
      case "q8" -> VectorUtil.ggufQ8_0Q8_0BatchedMatmulRows(w, b, r, c, from, to, out, a);
      case "block-q8" ->
          VectorUtil.ggufQ8_0Q8_0BlockMajorBatchedMatmulRows(w, b, r, c, from, to, out, a);
      default -> throw new AssertionError(format);
    }
  }
}
