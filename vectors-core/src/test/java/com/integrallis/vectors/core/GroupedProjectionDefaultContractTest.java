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
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Uses exactly representable integer fixtures so dense algebra is independent of SIMD folding. */
class GroupedProjectionDefaultContractTest {
  private static final ValueLayout.OfShort HALF =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @ParameterizedTest
  @ValueSource(ints = {32, 64, 256})
  void groupedQ8DefaultsPreserveAllMatrixBoundaries(int cols) {
    int[] rows = {2, 3, 5};
    int batches = 3;
    float[] input = input(batches, cols);
    MemorySegment[] weights = new MemorySegment[3];
    float[][] expected = new float[3][];
    for (int i = 0; i < 3; i++) {
      weights[i] = weights(rows[i], cols, i, false);
      expected[i] = dense(input, batches, rows[i], cols, i, false);
    }
    for (VectorUtilSupport provider :
        new VectorUtilSupport[] {
          VectorizationProvider.newScalarProvider(), VectorizationProvider.getInstance()
        }) {
      byte[] q = new byte[batches * cols];
      float[] scales = new float[batches * cols / 32];
      float[][] out = {
        guarded(batches * rows[0]), guarded(batches * rows[1]), guarded(batches * rows[2])
      };
      provider.ggufQ8_0Q8_0TripleBatchedMatmul(
          input,
          weights[0],
          rows[0],
          out[0],
          weights[1],
          rows[1],
          out[1],
          weights[2],
          rows[2],
          out[2],
          batches,
          cols,
          q,
          scales);
      for (int i = 0; i < 3; i++) verify(out[i], expected[i]);
      provider.ggufQ8_0Q8_0DualBatchedMatmul(
          input,
          weights[0],
          rows[0],
          out[0],
          weights[1],
          rows[1],
          out[1],
          batches,
          cols,
          q,
          scales);
      for (int i = 0; i < 2; i++) verify(out[i], expected[i]);
      float[][] single = {guarded(rows[0]), guarded(rows[1]), guarded(rows[2])};
      provider.ggufQ8_0Q8_0TripleMatVecDot(
          input,
          weights[0],
          rows[0],
          single[0],
          weights[1],
          rows[1],
          single[1],
          weights[2],
          rows[2],
          single[2],
          cols,
          q,
          scales);
      for (int i = 0; i < 3; i++) verify(single[i], Arrays.copyOf(expected[i], rows[i]));
      provider.ggufQ8_0Q8_0DualMatVecDot(
          input, weights[0], rows[0], single[0], weights[1], rows[1], single[1], cols, q, scales);
      for (int i = 0; i < 2; i++) verify(single[i], Arrays.copyOf(expected[i], rows[i]));
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {32, 64, 256})
  void groupedQ4DefaultsDecodeLowAndHighNibblesInLogicalOrder(int cols) {
    int[] rows = {2, 3, 5};
    float[] input = input(1, cols);
    MemorySegment[] weights = new MemorySegment[3];
    float[][] expected = new float[3][];
    for (int i = 0; i < 3; i++) {
      weights[i] = weights(rows[i], cols, i, true);
      expected[i] = dense(input, 1, rows[i], cols, i, true);
    }
    for (VectorUtilSupport provider :
        new VectorUtilSupport[] {
          VectorizationProvider.newScalarProvider(), VectorizationProvider.getInstance()
        })
      for (GgufQ4Kernel kernel : GgufQ4Kernel.values()) {
        byte[] q = new byte[cols];
        float[] scales = new float[cols / 32];
        int[] corrections = new int[cols / 16];
        float[][] out = {guarded(rows[0]), guarded(rows[1]), guarded(rows[2])};
        provider.ggufQ4_0Q8_0TripleMatVecDot(
            input,
            weights[0],
            rows[0],
            out[0],
            weights[1],
            rows[1],
            out[1],
            weights[2],
            rows[2],
            out[2],
            cols,
            q,
            scales,
            corrections,
            kernel);
        for (int i = 0; i < 3; i++) verify(out[i], expected[i]);
        provider.ggufQ4_0Q8_0DualMatVecDot(
            input,
            weights[0],
            rows[0],
            out[0],
            weights[1],
            rows[1],
            out[1],
            cols,
            q,
            scales,
            corrections,
            kernel);
        for (int i = 0; i < 2; i++) verify(out[i], expected[i]);
      }
  }

  @ParameterizedTest
  @ValueSource(ints = {1, 3, 7, 16, 37, 129})
  void executionCopyPreservesOwnedWeightsAndScalarTails(int cols) {
    int rows = 5;
    float[] weights = new float[rows * cols];
    for (int i = 0; i < weights.length; i++) weights[i] = (i % 13 - 6) * .25f;
    float[] original = weights.clone();
    F32ExecutionMatrix matrix = F32ExecutionMatrix.copyOf(weights, rows, cols, weights.length * 4L);
    Arrays.fill(weights, Float.NaN);
    for (int batches : new int[] {1, 2, 4, 5}) {
      float[] in = new float[batches * cols + 7];
      for (int i = 0; i < batches * cols; i++) in[i + 3] = (i % 9 - 4) * .5f;
      float[] out = guarded(batches * rows + 2);
      matrix.multiplyBatch(in, 3, batches, out, 2);
      assertThat(out[0]).isEqualTo(-991f);
      assertThat(out[1]).isEqualTo(-991f);
      for (int b = 0; b < batches; b++)
        for (int r = 0; r < rows; r++) {
          double sum = 0;
          for (int c = 0; c < cols; c++) sum += in[3 + b * cols + c] * original[r * cols + c];
          assertThat(out[2 + b * rows + r]).isEqualTo((float) sum);
        }
      assertThat(Arrays.copyOfRange(out, 2 + batches * rows, out.length)).containsOnly(-991f);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {256, 512})
  void groupedKQuantDefaultsKeepCorrectionSumsAndProjectionRowsSeparate(int cols) {
    int[] rows = {2, 3, 5};
    int batches = 3;
    float[] input = input(batches, cols);
    for (int b = 0; b < batches; b++)
      for (int c = 0; c < cols; c += 256) input[b * cols + c] = -127;
    MemorySegment[] weights = new MemorySegment[3];
    float[][] expected = new float[3][];
    for (int i = 0; i < 3; i++) {
      weights[i] = kWeights(rows[i], cols, i);
      expected[i] = dense(input, batches, rows[i], cols, i, true);
    }
    for (VectorUtilSupport provider :
        new VectorUtilSupport[] {
          VectorizationProvider.newScalarProvider(), VectorizationProvider.getInstance()
        }) {
      byte[] q = new byte[batches * cols];
      float[] scales = new float[batches * cols / 256];
      short[] sums = new short[batches * cols / 16];
      float[][] out = {guarded(batches * rows[0]), guarded(batches * rows[1])};
      provider.ggufQ4_KQ8_KDualBatchedMatmul(
          input,
          weights[0],
          rows[0],
          out[0],
          weights[1],
          rows[1],
          out[1],
          batches,
          cols,
          q,
          scales,
          sums);
      verify(out[0], expected[0]);
      verify(out[1], expected[1]);
      assertThat(scales).containsOnly(1f);
      float[][] single = {guarded(rows[0]), guarded(rows[1]), guarded(rows[2])};
      provider.ggufQ4_KQ8_KTripleMatVecDot(
          input,
          weights[0],
          rows[0],
          single[0],
          weights[1],
          rows[1],
          single[1],
          weights[2],
          rows[2],
          single[2],
          cols,
          q,
          scales,
          sums);
      for (int i = 0; i < 3; i++) verify(single[i], Arrays.copyOf(expected[i], rows[i]));
      provider.ggufQ4_KQ8_KDualMatVecDot(
          input,
          weights[0],
          rows[0],
          single[0],
          weights[1],
          rows[1],
          single[1],
          cols,
          q,
          scales,
          sums);
      for (int i = 0; i < 2; i++) verify(single[i], Arrays.copyOf(expected[i], rows[i]));
    }
  }

  private static MemorySegment kWeights(int rows, int cols, int matrix) {
    byte[] data = new byte[rows * cols / 256 * 144];
    MemorySegment w = MemorySegment.ofArray(data);
    for (int row = 0; row < rows; row++)
      for (int block = 0; block < cols / 256; block++) {
        int at = (row * (cols / 256) + block) * 144;
        w.set(HALF, at, Float.floatToFloat16(1f));
        w.set(HALF, at + 2, Float.floatToFloat16(8f));
        Arrays.fill(data, at + 4, at + 12, (byte) 1);
        Arrays.fill(data, at + 12, at + 16, (byte) 17);
        for (int group = 0; group < 8; group++)
          for (int c = 0; c < 32; c++) {
            int q = coefficient(matrix, row, block * 256 + group * 32 + c, true) + 8;
            data[at + 16 + (group / 2) * 32 + c] |= (byte) (q << (4 * (group % 2)));
          }
      }
    return w;
  }

  private static float[] input(int batches, int cols) {
    float[] x = new float[batches * cols];
    for (int b = 0; b < batches; b++)
      for (int c = 0; c < cols; c++)
        x[b * cols + c] = c % 32 == 0 ? 127 : (b * 3 + c * 5) % 61 - 30;
    return x;
  }

  private static int coefficient(int matrix, int row, int col, boolean q4) {
    return (matrix * 7 + row * 3 + col * 5) % (q4 ? 16 : 63) - (q4 ? 8 : 31);
  }

  private static MemorySegment weights(int rows, int cols, int matrix, boolean q4) {
    int size = q4 ? 18 : 34;
    byte[] data = new byte[rows * cols / 32 * size];
    MemorySegment w = MemorySegment.ofArray(data);
    for (int row = 0; row < rows; row++)
      for (int block = 0; block < cols / 32; block++) {
        int offset = (row * (cols / 32) + block) * size;
        w.set(HALF, offset, Float.floatToFloat16(1f));
        if (q4)
          for (int c = 0; c < 16; c++)
            data[offset + 2 + c] =
                (byte)
                    ((coefficient(matrix, row, block * 32 + c, true) + 8)
                        | ((coefficient(matrix, row, block * 32 + c + 16, true) + 8) << 4));
        else
          for (int c = 0; c < 32; c++)
            data[offset + 2 + c] = (byte) coefficient(matrix, row, block * 32 + c, false);
      }
    return w;
  }

  private static float[] dense(float[] x, int batches, int rows, int cols, int matrix, boolean q4) {
    float[] y = new float[batches * rows];
    for (int b = 0; b < batches; b++)
      for (int r = 0; r < rows; r++) {
        long sum = 0;
        for (int c = 0; c < cols; c++)
          sum += (long) x[b * cols + c] * coefficient(matrix, r, c, q4);
        y[b * rows + r] = sum;
      }
    return y;
  }

  private static float[] guarded(int n) {
    float[] x = new float[n + 3];
    Arrays.fill(x, -991f);
    return x;
  }

  private static void verify(float[] x, float[] expected) {
    assertThat(Arrays.copyOf(x, expected.length)).containsExactly(expected);
    assertThat(Arrays.copyOfRange(x, expected.length, x.length)).containsOnly(-991f);
  }
}
