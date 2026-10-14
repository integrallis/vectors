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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Checks that public batch validation is transactional for every supported packed format. */
class PublicQuantizedBatchBoundaryTest {
  static Stream<Arguments> malformedInputs() {
    Stream.Builder<Arguments> cases = Stream.builder();
    for (String format : new String[] {"q4_0", "q5_0", "q8_0", "q4_k", "q5_k", "q6_k"})
      for (String fault :
          new String[] {
            "batch",
            "rows",
            "columns",
            "query",
            "output",
            "weights",
            "quants",
            "scales",
            "null-query",
            "null-weights",
            "null-quants",
            "null-scales"
          }) cases.add(Arguments.of(format, fault));
    return cases.build();
  }

  @ParameterizedTest(name = "{0} rejects {1} before writes")
  @MethodSource("malformedInputs")
  void malformedBatchCannotPartiallyOverwriteCallerScratch(String format, String fault) {
    int columns = 256, batches = 2, rows = 2;
    int block = format.endsWith("_k") ? 256 : 32;
    int bytes =
        switch (format) {
          case "q4_0" -> 18;
          case "q5_0" -> 22;
          case "q8_0" -> 34;
          case "q4_k" -> 144;
          case "q5_k" -> 176;
          case "q6_k" -> 210;
          default -> throw new AssertionError(format);
        };
    float[] input = new float[batches * columns];
    Arrays.fill(input, 127f);
    float[] out = new float[batches * rows];
    Arrays.fill(out, 73f);
    byte[] quants = new byte[batches * columns];
    Arrays.fill(quants, (byte) 29);
    float[] scales = new float[batches * columns / block];
    Arrays.fill(scales, 31f);
    MemorySegment weight = MemorySegment.ofArray(new byte[rows * columns / block * bytes]);
    int[] corrections = new int[batches * columns / 16];
    Arrays.fill(corrections, 17);
    short[] sums = new short[batches * columns / 16];
    Arrays.fill(sums, (short) 19);
    float[] passedInput =
        fault.equals("query")
            ? Arrays.copyOf(input, input.length - 1)
            : fault.equals("null-query") ? null : input;
    float[] passedOut = fault.equals("output") ? Arrays.copyOf(out, out.length - 1) : out;
    byte[] passedQuants =
        fault.equals("quants")
            ? Arrays.copyOf(quants, quants.length - 1)
            : fault.equals("null-quants") ? null : quants;
    float[] passedScales =
        fault.equals("scales")
            ? Arrays.copyOf(scales, scales.length - 1)
            : fault.equals("null-scales") ? null : scales;
    MemorySegment passedWeight =
        fault.equals("weights")
            ? weight.asSlice(0, weight.byteSize() - 1)
            : fault.equals("null-weights") ? null : weight;
    int passedBatch = fault.equals("batch") ? 0 : batches;
    int passedRows = fault.equals("rows") ? 0 : rows;
    int passedColumns = fault.equals("columns") ? columns - 1 : columns;
    assertThatThrownBy(
            () ->
                invoke(
                    format,
                    passedInput,
                    passedWeight,
                    passedBatch,
                    passedRows,
                    passedColumns,
                    passedOut,
                    passedQuants,
                    passedScales,
                    corrections,
                    sums))
        .isInstanceOf(
            fault.startsWith("null-")
                ? NullPointerException.class
                : IllegalArgumentException.class);
    assertThat(out).containsOnly(73f);
    assertThat(passedOut).containsOnly(73f);
    assertThat(quants).containsOnly((byte) 29);
    if (passedQuants != null) assertThat(passedQuants).containsOnly((byte) 29);
    assertThat(scales).containsOnly(31f);
    if (passedScales != null) assertThat(passedScales).containsOnly(31f);
    assertThat(corrections).containsOnly(17);
    assertThat(sums).containsOnly((short) 19);
  }

  private static void invoke(
      String format,
      float[] input,
      MemorySegment w,
      int batches,
      int rows,
      int cols,
      float[] out,
      byte[] q,
      float[] scale,
      int[] correction,
      short[] sum) {
    switch (format) {
      case "q4_0" ->
          VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
              input, w, batches, rows, cols, out, q, scale, correction);
      case "q5_0" ->
          VectorUtil.ggufQ5_0Q8_0BatchedMatmul(input, w, batches, rows, cols, out, q, scale);
      case "q8_0" ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmul(input, w, batches, rows, cols, out, q, scale);
      case "q4_k" ->
          VectorUtil.ggufQ4_KQ8_KBatchedMatmul(input, w, batches, rows, cols, out, q, scale, sum);
      case "q5_k" ->
          VectorUtil.ggufQ5_KQ8_KBatchedMatmul(input, w, batches, rows, cols, out, q, scale, sum);
      case "q6_k" ->
          VectorUtil.ggufQ6_KQ8_KBatchedMatmul(input, w, batches, rows, cols, out, q, scale);
      default -> throw new AssertionError(format);
    }
  }
}
