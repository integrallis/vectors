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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PackedInt4GroupMatrixTest {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Test
  void rowMajorKernelMatchesAnIndependentDequantizedReference() {
    int rows = 3;
    int columns = 64;
    int groupSize = 32;
    byte[] packed = new byte[rows * columns / 2];
    byte[] scaleBytes = new byte[rows * (columns / groupSize) * Short.BYTES];
    float[] input = new float[columns];
    float[] expected = new float[rows];
    float[] actual = new float[rows];

    for (int column = 0; column < columns; column++) {
      input[column] = (column - 27) * 0.03125f;
    }
    for (int row = 0; row < rows; row++) {
      for (int group = 0; group < columns / groupSize; group++) {
        float scale = (row + 1) * (group + 1) * 0.125f;
        putScale(scaleBytes, row * 2 + group, scale);
        for (int offset = 0; offset < groupSize; offset++) {
          int column = group * groupSize + offset;
          int quant = ((row * 5 + column * 3) & 15) - 8;
          putNibble(packed, row * columns + column, quant);
          expected[row] += scale * quant * input[column];
        }
      }
    }

    MemorySegment weights = MemorySegment.ofArray(packed);
    MemorySegment scales = MemorySegment.ofArray(scaleBytes);
    VectorizationProvider.newScalarProvider()
        .packedInt4GroupMatVec(input, weights, scales, rows, columns, groupSize, actual);
    assertThat(actual).containsExactly(expected);

    new PanamaVectorUtilSupport()
        .packedInt4GroupMatVec(input, weights, scales, rows, columns, groupSize, actual);
    for (int row = 0; row < rows; row++) {
      assertThat(actual[row]).isCloseTo(expected[row], within(2.0e-5f));
    }
  }

  @Test
  void rightHandKernelMatchesAnIndependentDequantizedReference() {
    int inputs = 5;
    int outputs = 64;
    int groupSize = 32;
    byte[] packed = new byte[inputs * outputs / 2];
    byte[] scaleBytes = new byte[inputs * (outputs / groupSize) * Short.BYTES];
    float[] input = {0.25f, -0.5f, 1.25f, 0.75f, -0.125f};
    float[] expected = new float[outputs];
    float[] actual = new float[outputs];

    for (int inputIndex = 0; inputIndex < inputs; inputIndex++) {
      for (int group = 0; group < outputs / groupSize; group++) {
        float scale = (inputIndex + 1) * (group + 2) * 0.0625f;
        putScale(scaleBytes, inputIndex * 2 + group, scale);
        for (int offset = 0; offset < groupSize; offset++) {
          int output = group * groupSize + offset;
          int quant = ((inputIndex * 7 + output * 5) & 15) - 8;
          putNibble(packed, inputIndex * outputs + output, quant);
          expected[output] += input[inputIndex] * scale * quant;
        }
      }
    }

    MemorySegment weights = MemorySegment.ofArray(packed);
    MemorySegment scales = MemorySegment.ofArray(scaleBytes);
    VectorizationProvider.newScalarProvider()
        .packedInt4GroupRightMatVec(input, weights, scales, inputs, outputs, groupSize, actual);
    assertThat(actual).containsExactly(expected);

    new PanamaVectorUtilSupport()
        .packedInt4GroupRightMatVec(input, weights, scales, inputs, outputs, groupSize, actual);
    for (int output = 0; output < outputs; output++) {
      assertThat(actual[output]).isCloseTo(expected[output], within(2.0e-5f));
    }
  }

  @Test
  void rowMajorBatchReusesWeightsAcrossIndependentInputRows() {
    int batchSize = 3;
    int rows = 3;
    int columns = 64;
    int groupSize = 32;
    byte[] packed = new byte[rows * columns / 2];
    byte[] scaleBytes = new byte[rows * (columns / groupSize) * Short.BYTES];
    float[] input = new float[batchSize * columns];
    float[] expected = new float[batchSize * rows];
    float[] actual = new float[batchSize * rows];

    for (int batch = 0; batch < batchSize; batch++) {
      for (int column = 0; column < columns; column++) {
        input[batch * columns + column] = (batch + 1) * (column - 27) * 0.03125f;
      }
    }
    for (int row = 0; row < rows; row++) {
      for (int group = 0; group < columns / groupSize; group++) {
        float scale = (row + 1) * (group + 1) * 0.125f;
        putScale(scaleBytes, row * 2 + group, scale);
        for (int offset = 0; offset < groupSize; offset++) {
          int column = group * groupSize + offset;
          int quant = ((row * 5 + column * 3) & 15) - 8;
          putNibble(packed, row * columns + column, quant);
          for (int batch = 0; batch < batchSize; batch++) {
            expected[batch * rows + row] += scale * quant * input[batch * columns + column];
          }
        }
      }
    }

    MemorySegment weights = MemorySegment.ofArray(packed);
    MemorySegment scales = MemorySegment.ofArray(scaleBytes);
    new PanamaVectorUtilSupport()
        .packedInt4GroupMatVecBatch(
            input, batchSize, weights, scales, rows, columns, groupSize, actual);
    for (int index = 0; index < actual.length; index++) {
      assertThat(actual[index]).isCloseTo(expected[index], within(2.0e-5f));
    }
  }

  @Test
  void rightHandBatchReusesWeightsAcrossIndependentInputRows() {
    int batchSize = 3;
    int inputs = 5;
    int outputs = 64;
    int groupSize = 32;
    byte[] packed = new byte[inputs * outputs / 2];
    byte[] scaleBytes = new byte[inputs * (outputs / groupSize) * Short.BYTES];
    float[] input = new float[batchSize * inputs];
    float[] expected = new float[batchSize * outputs];
    float[] actual = new float[batchSize * outputs];

    for (int batch = 0; batch < batchSize; batch++) {
      for (int inputIndex = 0; inputIndex < inputs; inputIndex++) {
        input[batch * inputs + inputIndex] = (batch + 1) * (inputIndex - 2) * 0.125f;
      }
    }
    for (int inputIndex = 0; inputIndex < inputs; inputIndex++) {
      for (int group = 0; group < outputs / groupSize; group++) {
        float scale = (inputIndex + 1) * (group + 2) * 0.0625f;
        putScale(scaleBytes, inputIndex * 2 + group, scale);
        for (int offset = 0; offset < groupSize; offset++) {
          int output = group * groupSize + offset;
          int quant = ((inputIndex * 7 + output * 5) & 15) - 8;
          putNibble(packed, inputIndex * outputs + output, quant);
          for (int batch = 0; batch < batchSize; batch++) {
            expected[batch * outputs + output] +=
                input[batch * inputs + inputIndex] * scale * quant;
          }
        }
      }
    }

    MemorySegment weights = MemorySegment.ofArray(packed);
    MemorySegment scales = MemorySegment.ofArray(scaleBytes);
    new PanamaVectorUtilSupport()
        .packedInt4GroupRightMatVecBatch(
            input, batchSize, weights, scales, inputs, outputs, groupSize, actual);
    for (int index = 0; index < actual.length; index++) {
      assertThat(actual[index]).isCloseTo(expected[index], within(2.0e-5f));
    }
  }

  @Test
  void publicFacadeRejectsMalformedGeometryBeforeDispatch() {
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                VectorUtil.packedInt4GroupMatVec(
                    new float[31],
                    MemorySegment.ofArray(new byte[16]),
                    MemorySegment.ofArray(new byte[2]),
                    1,
                    31,
                    32,
                    new float[1]))
        .withMessageContaining("columns");
  }

  private static void putScale(byte[] bytes, int index, float value) {
    MemorySegment.ofArray(bytes)
        .set(LE_SHORT, (long) index * Short.BYTES, Float.floatToFloat16(value));
  }

  private static void putNibble(byte[] packed, int logicalIndex, int signedValue) {
    int index = logicalIndex >>> 1;
    int nibble = signedValue & 15;
    if ((logicalIndex & 1) == 0) {
      packed[index] = (byte) ((packed[index] & 0xf0) | nibble);
    } else {
      packed[index] = (byte) ((packed[index] & 0x0f) | (nibble << 4));
    }
  }
}
