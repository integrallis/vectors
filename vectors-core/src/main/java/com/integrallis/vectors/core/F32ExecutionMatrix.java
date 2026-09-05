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
import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * An owned row-major F32 execution matrix prepared from a more compact serialized representation.
 *
 * <p>The source decoder remains responsible for interpreting its wire format. This class owns an
 * immutable copy of the resulting values and provides batch-major multiplication optimized for
 * repeated inference. It is useful when widening weights once is faster than decoding compact
 * values on every projection.
 */
public final class F32ExecutionMatrix {

  private static final VectorSpecies<Float> SPECIES = PanamaVectorUtilSupport.FLOAT_SPECIES;

  private final int rows;
  private final int columns;
  private final long serializedByteCount;
  private final float[] weights;
  private final MemorySegment executionStorage;

  private F32ExecutionMatrix(
      int rows, int columns, long serializedByteCount, float[] rowMajorWeights) {
    this.rows = rows;
    this.columns = columns;
    this.serializedByteCount = serializedByteCount;
    this.weights = rowMajorWeights;
    this.executionStorage = MemorySegment.ofArray(rowMajorWeights).asReadOnly();
  }

  /** Creates an owned execution copy of an exactly sized row-major matrix. */
  public static F32ExecutionMatrix copyOf(
      float[] rowMajorWeights, int rows, int columns, long serializedByteCount) {
    Objects.requireNonNull(rowMajorWeights, "rowMajorWeights");
    requirePositive(rows, "rows");
    requirePositive(columns, "columns");
    requirePositive(serializedByteCount, "serializedByteCount");
    int expected = Math.multiplyExact(rows, columns);
    if (rowMajorWeights.length != expected) {
      throw new IllegalArgumentException(
          "rowMajorWeights length must be " + expected + "; got " + rowMajorWeights.length);
    }
    return new F32ExecutionMatrix(rows, columns, serializedByteCount, rowMajorWeights.clone());
  }

  /** Returns the number of output rows. */
  public int rows() {
    return rows;
  }

  /** Returns the number of input columns. */
  public int columns() {
    return columns;
  }

  /** Returns the compact source representation's byte count. */
  public long serializedByteCount() {
    return serializedByteCount;
  }

  /** Returns the bytes retained by the F32 execution layout. */
  public long expandedByteCount() {
    return (long) weights.length * Float.BYTES;
  }

  /** Returns the execution-layout size divided by the serialized size. */
  public double expansionRatio() {
    return (double) expandedByteCount() / serializedByteCount;
  }

  /** Multiplies complete batch-major F32 input and output arrays. */
  public void multiplyBatch(float[] input, int batchSize, float[] output) {
    multiplyBatch(input, 0, batchSize, output, 0);
  }

  /** Multiplies consecutive batch-major inputs into consecutive batch-major output rows. */
  public void multiplyBatch(
      float[] input, int inputOffset, int batchSize, float[] output, int outputOffset) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    requirePositive(batchSize, "batchSize");
    int inputElements = Math.multiplyExact(batchSize, columns);
    int outputElements = Math.multiplyExact(batchSize, rows);
    requireArrayRange(input.length, inputOffset, inputElements, "input");
    requireArrayRange(output.length, outputOffset, outputElements, "output");

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

  private static void requirePositive(long value, String name) {
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
