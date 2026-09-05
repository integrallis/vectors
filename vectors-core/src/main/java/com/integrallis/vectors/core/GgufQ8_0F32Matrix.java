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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * A row-major GGUF Q8_0 matrix expanded once to an F32 execution layout.
 *
 * <p>The expanded layout trades memory for F32 matrix-multiplication throughput. It owns the
 * decoded weights and does not retain the source segment, so it remains valid after a mapped GGUF
 * tensor is closed. Inputs and outputs are batch-major: each input row has {@link #columns()}
 * elements and each output row has {@link #rows()} elements.
 */
public final class GgufQ8_0F32Matrix {

  private static final int BLOCK_ELEMENTS = VectorUtilSupport.GGUF_Q_BLOCK_SIZE;
  private static final int BLOCK_BYTES = VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES;
  private static final VectorSpecies<Float> SPECIES = PanamaVectorUtilSupport.FLOAT_SPECIES;

  private final int rows;
  private final int columns;
  private final long serializedByteCount;
  private final float[] weights;
  private final MemorySegment executionStorage;

  private GgufQ8_0F32Matrix(int rows, int columns, long serializedByteCount, float[] weights) {
    this.rows = rows;
    this.columns = columns;
    this.serializedByteCount = serializedByteCount;
    this.weights = weights;
    executionStorage = MemorySegment.ofArray(weights).asReadOnly();
  }

  /**
   * Expands an exactly sized, row-major GGUF Q8_0 matrix into an owned F32 execution layout.
   *
   * @param q8Weights GGUF Q8_0 blocks, with one FP16 scale followed by 32 signed-byte quants
   * @param rows logical output rows
   * @param columns logical input columns, which must be a positive multiple of 32
   * @return a matrix independent of the source segment's lifetime
   */
  public static GgufQ8_0F32Matrix from(MemorySegment q8Weights, int rows, int columns) {
    Objects.requireNonNull(q8Weights, "q8Weights");
    requirePositive(rows, "rows");
    requirePositive(columns, "columns");
    if (columns % BLOCK_ELEMENTS != 0) {
      throw new IllegalArgumentException("columns must be a multiple of 32: " + columns);
    }

    int blocksPerRow = columns / BLOCK_ELEMENTS;
    long serializedBytes = Math.multiplyExact((long) rows * blocksPerRow, BLOCK_BYTES);
    if (q8Weights.byteSize() != serializedBytes) {
      throw new IllegalArgumentException(
          "Q8_0 matrix requires "
              + serializedBytes
              + " bytes for "
              + rows
              + "x"
              + columns
              + "; got "
              + q8Weights.byteSize());
    }
    long expandedElements = Math.multiplyExact((long) rows, columns);
    if (expandedElements > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "expanded F32 matrix exceeds the maximum Java array length: " + expandedElements);
    }

    float[] decoded = new float[(int) expandedElements];
    for (int row = 0; row < rows; row++) {
      for (int block = 0; block < blocksPerRow; block++) {
        long blockOffset = ((long) row * blocksPerRow + block) * BLOCK_BYTES;
        float scale =
            Float.float16ToFloat(q8Weights.get(VectorUtilSupport.GGUF_LE_SHORT, blockOffset));
        int decodedOffset = row * columns + block * BLOCK_ELEMENTS;
        for (int lane = 0; lane < BLOCK_ELEMENTS; lane++) {
          decoded[decodedOffset + lane] =
              scale * q8Weights.get(ValueLayout.JAVA_BYTE, blockOffset + Short.BYTES + lane);
        }
      }
    }
    return new GgufQ8_0F32Matrix(rows, columns, serializedBytes, decoded);
  }

  /** Returns the number of output rows. */
  public int rows() {
    return rows;
  }

  /** Returns the number of input columns. */
  public int columns() {
    return columns;
  }

  /** Returns the byte count of the source GGUF Q8_0 representation. */
  public long serializedByteCount() {
    return serializedByteCount;
  }

  /** Returns the byte count retained by the expanded F32 execution layout. */
  public long expandedByteCount() {
    return (long) weights.length * Float.BYTES;
  }

  /** Returns {@link #expandedByteCount()} divided by {@link #serializedByteCount()}. */
  public double expansionRatio() {
    return (double) expandedByteCount() / serializedByteCount;
  }

  /** Multiplies complete batch-major F32 input and output arrays. */
  public void multiplyBatch(float[] input, int batchSize, float[] output) {
    multiplyBatch(input, 0, batchSize, output, 0);
  }

  /**
   * Multiplies consecutive batch-major F32 activations into consecutive batch-major output rows.
   */
  public void multiplyBatch(
      float[] input, int inputOffset, int batchSize, float[] output, int outputOffset) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    requirePositive(batchSize, "batchSize");
    int inputElements = Math.multiplyExact(batchSize, columns);
    int outputElements = Math.multiplyExact(batchSize, rows);
    requireArrayRange(input.length, inputOffset, inputElements, "input");
    requireArrayRange(output.length, outputOffset, outputElements, "output");

    // Parallelization is by output row. Include all batch activations in the work estimate so a
    // short, wide, high-batch projection does not accidentally take the serial path.
    GgufParallelSupport.forEachRow(
        executionStorage,
        rows,
        inputElements,
        row -> multiplyRow(input, inputOffset, batchSize, output, outputOffset, row));
  }

  private void multiplyRow(
      float[] input, int inputOffset, int batchSize, float[] output, int outputOffset, int row) {
    int weightOffset = row * columns;
    int batch = 0;
    for (; batch + 3 < batchSize; batch += 4) {
      int input0 = inputOffset + batch * columns;
      int input1 = input0 + columns;
      int input2 = input1 + columns;
      int input3 = input2 + columns;
      FloatVector sum0 = FloatVector.zero(SPECIES);
      FloatVector sum1 = FloatVector.zero(SPECIES);
      FloatVector sum2 = FloatVector.zero(SPECIES);
      FloatVector sum3 = FloatVector.zero(SPECIES);
      int column = 0;
      int vectorBound = SPECIES.loopBound(columns);
      for (; column < vectorBound; column += SPECIES.length()) {
        FloatVector weight = FloatVector.fromArray(SPECIES, weights, weightOffset + column);
        sum0 =
            PanamaVectorUtilSupport.fma(
                weight, FloatVector.fromArray(SPECIES, input, input0 + column), sum0);
        sum1 =
            PanamaVectorUtilSupport.fma(
                weight, FloatVector.fromArray(SPECIES, input, input1 + column), sum1);
        sum2 =
            PanamaVectorUtilSupport.fma(
                weight, FloatVector.fromArray(SPECIES, input, input2 + column), sum2);
        sum3 =
            PanamaVectorUtilSupport.fma(
                weight, FloatVector.fromArray(SPECIES, input, input3 + column), sum3);
      }
      float scalar0 = sum0.reduceLanes(VectorOperators.ADD);
      float scalar1 = sum1.reduceLanes(VectorOperators.ADD);
      float scalar2 = sum2.reduceLanes(VectorOperators.ADD);
      float scalar3 = sum3.reduceLanes(VectorOperators.ADD);
      for (; column < columns; column++) {
        float weight = weights[weightOffset + column];
        scalar0 = MathUtil.fma(weight, input[input0 + column], scalar0);
        scalar1 = MathUtil.fma(weight, input[input1 + column], scalar1);
        scalar2 = MathUtil.fma(weight, input[input2 + column], scalar2);
        scalar3 = MathUtil.fma(weight, input[input3 + column], scalar3);
      }
      int result = outputOffset + batch * rows + row;
      output[result] = scalar0;
      output[result + rows] = scalar1;
      output[result + 2 * rows] = scalar2;
      output[result + 3 * rows] = scalar3;
    }

    for (; batch < batchSize; batch++) {
      int activationOffset = inputOffset + batch * columns;
      FloatVector sum = FloatVector.zero(SPECIES);
      int column = 0;
      int vectorBound = SPECIES.loopBound(columns);
      for (; column < vectorBound; column += SPECIES.length()) {
        FloatVector weight = FloatVector.fromArray(SPECIES, weights, weightOffset + column);
        sum =
            PanamaVectorUtilSupport.fma(
                weight, FloatVector.fromArray(SPECIES, input, activationOffset + column), sum);
      }
      float scalar = sum.reduceLanes(VectorOperators.ADD);
      for (; column < columns; column++) {
        scalar =
            MathUtil.fma(weights[weightOffset + column], input[activationOffset + column], scalar);
      }
      output[outputOffset + batch * rows + row] = scalar;
    }
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }

  private static void requireArrayRange(int length, int offset, int count, String name) {
    if (offset < 0 || count < 0 || offset > length - count) {
      throw new IllegalArgumentException(
          name + " range [" + offset + ", " + ((long) offset + count) + ") exceeds " + length);
    }
  }
}
