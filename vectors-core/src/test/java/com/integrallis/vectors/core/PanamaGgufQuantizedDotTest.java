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

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PanamaGgufQuantizedDotTest {

  @Test
  void panamaProviderOwnsQ4_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ4_0Q8_0MatVecDot");
  }

  @Test
  void panamaProviderOwnsQ8_0Q8_0Kernel() {
    assertThat(PanamaVectorUtilSupport.class.getDeclaredMethods())
        .extracting(Method::getName)
        .contains("ggufQ8_0Q8_0MatVecDot");
  }

  @Test
  void q4_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x5148L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 18];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 18) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ4_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ4_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).containsExactly(expected);
  }

  @Test
  void q8_0Q8_0KernelMatchesScalarReferenceAcrossRowsAndBlocks() {
    int rows = 512;
    int cols = 2048;
    Random random = new Random(0x5188L);
    float[] query = new float[cols];
    for (int index = 0; index < cols; index++) {
      query[index] = random.nextFloat() * 4.0f - 2.0f;
    }

    byte[] weights = new byte[rows * (cols / 32) * 34];
    random.nextBytes(weights);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int offset = 0; offset < weights.length; offset += 34) {
      float scale = random.nextFloat() * 0.05f + 0.001f;
      buffer.putShort(offset, Float.floatToFloat16(scale));
    }

    float[] expected = new float[rows];
    float[] actual = new float[rows];
    byte[] expectedQuants = new byte[cols];
    byte[] actualQuants = new byte[cols];
    float[] expectedScales = new float[cols / 32];
    float[] actualScales = new float[cols / 32];
    MemorySegment weightSegment = MemorySegment.ofArray(weights);

    new ScalarVectorUtilSupport()
        .ggufQ8_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, expected, expectedQuants, expectedScales);
    new PanamaVectorUtilSupport()
        .ggufQ8_0Q8_0MatVecDot(
            query, weightSegment, rows, cols, actual, actualQuants, actualScales);

    assertThat(actualQuants).containsExactly(expectedQuants);
    assertThat(actualScales).containsExactly(expectedScales);
    assertThat(actual).containsExactly(expected);
  }
}
