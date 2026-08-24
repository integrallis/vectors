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
import java.nio.ByteOrder;
import java.util.Objects;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;

/** A row-major matrix stored as little-endian IEEE bfloat16 values. */
public final class BFloat16Matrix {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final int BYTES_PER_VALUE = Short.BYTES;
  private static final int SEQUENTIAL_ROW_THRESHOLD = 8192;
  private static final VectorUtilSupport VECTOR_SUPPORT = VectorizationProvider.getInstance();
  private static final VectorShuffle<Float> EVEN_INPUTS =
      VectorShuffle.makeUnzip(PanamaVectorUtilSupport.FLOAT_SPECIES, 0);
  private static final VectorShuffle<Float> ODD_INPUTS =
      VectorShuffle.makeUnzip(PanamaVectorUtilSupport.FLOAT_SPECIES, 1);
  private static final int VALUES_PER_VECTOR =
      Math.multiplyExact(PanamaVectorUtilSupport.INT_SPECIES.length(), 2);

  private final MemorySegment data;
  private final int rows;
  private final int columns;

  private BFloat16Matrix(MemorySegment data, int rows, int columns) {
    this.data = data.asReadOnly();
    this.rows = rows;
    this.columns = columns;
  }

  /** Creates a zero-copy view over an exactly sized row-major bfloat16 matrix. */
  public static BFloat16Matrix of(MemorySegment data, int rows, int columns) {
    Objects.requireNonNull(data, "data");
    requirePositive(rows, "rows");
    requirePositive(columns, "columns");
    long expectedBytes =
        Math.multiplyExact(Math.multiplyExact((long) rows, columns), BYTES_PER_VALUE);
    if (data.byteSize() != expectedBytes) {
      throw new IllegalArgumentException(
          "bfloat16 matrix requires " + expectedBytes + " bytes; got " + data.byteSize());
    }
    return new BFloat16Matrix(data, rows, columns);
  }

  /** Returns the number of output rows. */
  public int rows() {
    return rows;
  }

  /** Returns the number of input columns. */
  public int columns() {
    return columns;
  }

  /** Multiplies this matrix by one F32 activation using F32 accumulation. */
  public void multiply(float[] input, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (input.length != columns) {
      throw new IllegalArgumentException(
          "input length must equal matrix columns " + columns + "; got " + input.length);
    }
    if (output.length < rows) {
      throw new IllegalArgumentException(
          "output length must be at least matrix rows " + rows + "; got " + output.length);
    }

    if (VECTOR_SUPPORT instanceof PanamaVectorUtilSupport) {
      vectorMultiply(input, data, rows, columns, output);
    } else {
      VECTOR_SUPPORT.bfloat16MatVecDot(input, data, rows, columns, output);
    }
  }

  /** Reads one stored matrix value as F32 without changing its exact bfloat16 value. */
  public float value(int row, int column) {
    if (row < 0 || row >= rows || column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException(
          "matrix coordinate (" + row + ", " + column + ") is outside " + rows + "x" + columns);
    }
    long offset = ((long) row * columns + column) * BYTES_PER_VALUE;
    return decode(data.get(LE_SHORT, offset));
  }

  /** Returns a zero-copy view over a contiguous range of rows. */
  public BFloat16Matrix rowSlice(int fromRow, int rowCount) {
    if (fromRow < 0 || rowCount <= 0 || fromRow > rows - rowCount) {
      throw new IndexOutOfBoundsException(
          "row range [" + fromRow + ", " + ((long) fromRow + rowCount) + ") exceeds " + rows);
    }
    long rowBytes = (long) columns * BYTES_PER_VALUE;
    return new BFloat16Matrix(
        data.asSlice(fromRow * rowBytes, rowCount * rowBytes), rowCount, columns);
  }

  private static float decode(short bits) {
    return Float.intBitsToFloat(Short.toUnsignedInt(bits) << Short.SIZE);
  }

  static void vectorMultiply(
      float[] input, MemorySegment weight, int rows, int columns, float[] output) {
    // Four interleaved rows reuse activations for ordinary transformer projections. Very tall
    // vocabulary matrices perform better as one sequential weight stream on both tested x86 hosts.
    if (rows >= SEQUENTIAL_ROW_THRESHOLD) {
      vectorMultiplySequential(input, weight, rows, columns, output);
    } else {
      vectorMultiplyFourRows(input, weight, rows, columns, output);
    }
  }

  private static void vectorMultiplyFourRows(
      float[] input, MemorySegment weight, int rows, int columns, float[] output) {
    int rowGroup = rows & ~3;
    int vectorBound = columns - columns % VALUES_PER_VECTOR;
    long rowBytes = (long) columns * Short.BYTES;
    for (int row = 0; row < rowGroup; row += 4) {
      long base0 = row * rowBytes;
      long base1 = base0 + rowBytes;
      long base2 = base1 + rowBytes;
      long base3 = base2 + rowBytes;
      FloatVector sum0 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      FloatVector sum1 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      FloatVector sum2 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      FloatVector sum3 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      for (int column = 0; column < vectorBound; column += VALUES_PER_VECTOR) {
        FloatVector first =
            FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, input, column);
        FloatVector second =
            FloatVector.fromArray(
                PanamaVectorUtilSupport.FLOAT_SPECIES,
                input,
                column + PanamaVectorUtilSupport.FLOAT_SPECIES.length());
        FloatVector evenInputs = first.rearrange(EVEN_INPUTS, second);
        FloatVector oddInputs = first.rearrange(ODD_INPUTS, second);
        long byteOffset = (long) column * Short.BYTES;
        sum0 = accumulate(weight, sum0, evenInputs, oddInputs, base0 + byteOffset);
        sum1 = accumulate(weight, sum1, evenInputs, oddInputs, base1 + byteOffset);
        sum2 = accumulate(weight, sum2, evenInputs, oddInputs, base2 + byteOffset);
        sum3 = accumulate(weight, sum3, evenInputs, oddInputs, base3 + byteOffset);
      }
      output[row] = finishRow(weight, sum0, input, columns, base0, vectorBound);
      output[row + 1] = finishRow(weight, sum1, input, columns, base1, vectorBound);
      output[row + 2] = finishRow(weight, sum2, input, columns, base2, vectorBound);
      output[row + 3] = finishRow(weight, sum3, input, columns, base3, vectorBound);
    }
    vectorMultiplyRows(input, weight, rowGroup, rows, columns, output);
  }

  private static void vectorMultiplySequential(
      float[] input, MemorySegment weight, int rows, int columns, float[] output) {
    vectorMultiplyRows(input, weight, 0, rows, columns, output);
  }

  private static void vectorMultiplyRows(
      float[] input, MemorySegment weight, int firstRow, int rows, int columns, float[] output) {
    int vectorBound = columns - columns % VALUES_PER_VECTOR;
    long rowBytes = (long) columns * Short.BYTES;
    for (int row = firstRow; row < rows; row++) {
      long rowOffset = row * rowBytes;
      FloatVector sum = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      for (int column = 0; column < vectorBound; column += VALUES_PER_VECTOR) {
        FloatVector first =
            FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, input, column);
        FloatVector second =
            FloatVector.fromArray(
                PanamaVectorUtilSupport.FLOAT_SPECIES,
                input,
                column + PanamaVectorUtilSupport.FLOAT_SPECIES.length());
        sum =
            accumulate(
                weight,
                sum,
                first.rearrange(EVEN_INPUTS, second),
                first.rearrange(ODD_INPUTS, second),
                rowOffset + (long) column * Short.BYTES);
      }
      output[row] = finishRow(weight, sum, input, columns, rowOffset, vectorBound);
    }
  }

  private static FloatVector accumulate(
      MemorySegment weight,
      FloatVector sum,
      FloatVector evenInputs,
      FloatVector oddInputs,
      long byteOffset) {
    IntVector packed =
        IntVector.fromMemorySegment(
            PanamaVectorUtilSupport.INT_SPECIES, weight, byteOffset, ByteOrder.LITTLE_ENDIAN);
    FloatVector evenWeights =
        packed.lanewise(VectorOperators.LSHL, Short.SIZE).reinterpretAsFloats();
    FloatVector oddWeights = packed.and(0xffff0000).reinterpretAsFloats();
    sum = PanamaVectorUtilSupport.fma(evenWeights, evenInputs, sum);
    return PanamaVectorUtilSupport.fma(oddWeights, oddInputs, sum);
  }

  private static float finishRow(
      MemorySegment weight,
      FloatVector vectorSum,
      float[] input,
      int columns,
      long rowOffset,
      int vectorBound) {
    float sum = vectorSum.reduceLanes(VectorOperators.ADD);
    for (int column = vectorBound; column < columns; column++) {
      float value = decode(weight.get(LE_SHORT, rowOffset + (long) column * Short.BYTES));
      sum = MathUtil.fma(input[column], value, sum);
    }
    return sum;
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive; got " + value);
    }
  }
}
