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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

class MatrixInputOutputOwnershipTest {
  @ParameterizedTest
  @CsvSource({
    "f32,0,0,1",
    "f32,0,31,1",
    "f32,4,0,1",
    "f32,0,60,2",
    "bf16,0,0,1",
    "bf16,0,31,1",
    "bf16,4,0,1",
    "bf16,0,60,2"
  })
  void overlappingRangesRejectBeforeMutation(String format, int inAt, int outAt, int batch) {
    float[] shared = new float[80];
    for (int i = 0; i < shared.length; i++) shared[i] = i * .25f - 2;
    float[] before = shared.clone();
    assertThatThrownBy(() -> multiply(format, shared, inAt, batch, shared, outAt))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("overlap");
    assertThat(shared).containsExactly(before);
  }

  @ParameterizedTest
  @CsvSource({"f32,0,64", "f32,10,0", "bf16,0,64", "bf16,10,0"})
  void disjointRangesInOneArrayRemainSupported(String format, int inAt, int outAt) {
    float[] shared = new float[76];
    Arrays.fill(shared, -1234);
    Arrays.fill(shared, inAt, inAt + 64, 1);
    float[] before = shared.clone();
    multiply(format, shared, inAt, 2, shared, outAt);
    for (int i = 0; i < shared.length; i++) {
      assertThat(shared[i]).isEqualTo(i >= outAt && i < outAt + 10 ? 32 : before[i]);
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"bf16", "mxfp4"})
  void directMultiplyRejectsInPlaceWithoutChangingInput(String format) {
    float[] input = new float[32];
    Arrays.fill(input, 1);
    float[] before = input.clone();
    assertThatThrownBy(
            () -> {
              if (format.equals("bf16")) bf16().multiply(input, input);
              else mxfp4().multiply(input, input);
            })
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("overlap");
    assertThat(input).containsExactly(before);
  }

  @ParameterizedTest
  @ValueSource(strings = {"f32", "bf16", "mxfp4"})
  void independentBuffersPreserveInputAndOutputCanaries(String format) {
    float[] input = new float[32];
    Arrays.fill(input, 1);
    float[] before = input.clone();
    float[] output = new float[7];
    Arrays.fill(output, -1234);
    if (format.equals("mxfp4")) mxfp4().multiply(input, output);
    else if (format.equals("bf16")) bf16().multiply(input, output);
    else f32().multiplyBatch(input, 1, output);
    assertThat(input).containsExactly(before);
    assertThat(output).containsExactly(32, 32, 32, 32, 32, -1234, -1234);
  }

  @ParameterizedTest
  @ValueSource(strings = {"f32", "bf16"})
  void invalidAliasedRangesFailBeforeMutation(String format) {
    float[] input = new float[64];
    Arrays.fill(input, 1);
    float[] before = input.clone();
    assertThatThrownBy(() -> multiply(format, input, Integer.MAX_VALUE, 1, input, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("input range");
    assertThat(input).containsExactly(before);
  }

  @Test
  void mxfp4QuantizationMakesOriginalInputSafeAsOutput() {
    var matrix = mxfp4();
    float[] input = new float[32];
    for (int i = 0; i < input.length; i++) input[i] = (i - 16) * .25f;
    float[] expected = new float[5];
    matrix.multiplyQ8(input, expected, GgufQ8_0Batch.allocate(1, 32));
    float[] before = input.clone();
    matrix.multiplyQ8(input, input, GgufQ8_0Batch.allocate(1, 32));
    assertThat(Arrays.copyOf(input, 5)).containsExactly(expected);
    assertThat(Arrays.copyOfRange(input, 5, 32)).containsExactly(Arrays.copyOfRange(before, 5, 32));
  }

  @Test
  void rotatedPreparationOwnsInputAndRemainsReusableWhenOriginalBecomesOutput() {
    var matrix =
        RotatedCodebookMatrix.of(
            MemorySegment.ofArray(new byte[4]),
            MemorySegment.ofArray(new byte[] {0, 60, 0, 60}),
            2,
            8,
            8,
            RotatedCodebookMatrix.Encoding.CQ2,
            new float[] {1, 2, 3, 4});
    float[] input = {1, 2, 3, 4, 5, 6, 7, 8};
    var prepared = matrix.prepare(input);
    float[] expected = new float[2];
    matrix.multiply(prepared, expected);
    Arrays.fill(input, -71);
    matrix.multiply(prepared, input);
    assertThat(Arrays.copyOf(input, 2)).containsExactly(expected);
    assertThat(Arrays.copyOfRange(input, 2, 8)).containsOnly(-71);
    float[] reused = new float[2];
    matrix.multiply(prepared, reused);
    assertThat(reused).containsExactly(expected);
  }

  private static void multiply(
      String format, float[] input, int inAt, int batch, float[] output, int outAt) {
    if (format.equals("f32")) f32().multiplyBatch(input, inAt, batch, output, outAt);
    else bf16().multiplyBatch(input, inAt, batch, output, outAt);
  }

  private static F32ExecutionMatrix f32() {
    float[] weights = new float[160];
    Arrays.fill(weights, 1);
    return F32ExecutionMatrix.copyOf(weights, 5, 32, 640);
  }

  private static BFloat16Matrix bf16() {
    ByteBuffer weights = ByteBuffer.allocate(320).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 160; i++) weights.putShort((short) 0x3f80);
    return BFloat16Matrix.of(MemorySegment.ofArray(weights.array()), 5, 32);
  }

  private static Mxfp4Matrix mxfp4() {
    byte[] weights = new byte[80];
    Arrays.fill(weights, (byte) 0x22);
    byte[] scales = new byte[5];
    Arrays.fill(scales, (byte) 127);
    return Mxfp4Matrix.of(MemorySegment.ofArray(weights), MemorySegment.ofArray(scales), 5, 32);
  }
}
