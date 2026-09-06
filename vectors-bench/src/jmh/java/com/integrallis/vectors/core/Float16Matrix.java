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
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;

/** Benchmark-only candidate for mapped little-endian IEEE 754 binary16 matrix execution. */
public final class Float16Matrix {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final int BYTES_PER_VALUE = Short.BYTES;
  private static final VectorShuffle<Float> EVEN_INPUTS =
      VectorShuffle.makeUnzip(PanamaVectorUtilSupport.FLOAT_SPECIES, 0);
  private static final VectorShuffle<Float> ODD_INPUTS =
      VectorShuffle.makeUnzip(PanamaVectorUtilSupport.FLOAT_SPECIES, 1);
  private static final int VALUES_PER_VECTOR =
      Math.multiplyExact(PanamaVectorUtilSupport.INT_SPECIES.length(), 2);

  private final MemorySegment data;
  private final int rows;
  private final int columns;

  private Float16Matrix(MemorySegment data, int rows, int columns) {
    this.data = data.asReadOnly();
    this.rows = rows;
    this.columns = columns;
  }

  /** Creates a zero-copy view over an exactly sized row-major binary16 matrix. */
  public static Float16Matrix of(MemorySegment data, int rows, int columns) {
    Objects.requireNonNull(data, "data");
    requirePositive(rows, "rows");
    requirePositive(columns, "columns");
    long expectedBytes =
        Math.multiplyExact(Math.multiplyExact((long) rows, columns), BYTES_PER_VALUE);
    if (data.byteSize() != expectedBytes) {
      throw new IllegalArgumentException(
          "float16 matrix requires " + expectedBytes + " bytes; got " + data.byteSize());
    }
    return new Float16Matrix(data, rows, columns);
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
    multiplyVectorRows(input, 0, data, rows, columns, output, 0);
  }

  /** Multiplies consecutive batch-major F32 activations without expanding the matrix. */
  public void multiplyBatch(
      float[] input, int inputOffset, int batchSize, float[] output, int outputOffset) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    requirePositive(batchSize, "batchSize");
    requireArrayRange(input.length, inputOffset, Math.multiplyExact(batchSize, columns), "input");
    requireArrayRange(output.length, outputOffset, Math.multiplyExact(batchSize, rows), "output");
    multiplyVectorBatch(input, inputOffset, batchSize, output, outputOffset);
  }

  /** Reads one stored matrix value as F32 without changing its exact binary16 value. */
  public float value(int row, int column) {
    if (row < 0 || row >= rows || column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException(
          "matrix coordinate (" + row + ", " + column + ") is outside " + rows + "x" + columns);
    }
    return valueAt((long) row * columns + column);
  }

  /** Returns a zero-copy view over a contiguous range of rows. */
  public Float16Matrix rowSlice(int fromRow, int rowCount) {
    if (fromRow < 0 || rowCount <= 0 || fromRow > rows - rowCount) {
      throw new IndexOutOfBoundsException(
          "row range [" + fromRow + ", " + ((long) fromRow + rowCount) + ") exceeds " + rows);
    }
    long rowBytes = (long) columns * BYTES_PER_VALUE;
    return new Float16Matrix(
        data.asSlice(fromRow * rowBytes, rowCount * rowBytes), rowCount, columns);
  }

  private static void multiplyVectorRows(
      float[] input,
      int inputOffset,
      MemorySegment weight,
      int rows,
      int columns,
      float[] output,
      int outputOffset) {
    int vectorBound = columns - columns % VALUES_PER_VECTOR;
    long rowBytes = (long) columns * BYTES_PER_VALUE;
    for (int row = 0; row < rows; row++) {
      long rowOffset = row * rowBytes;
      FloatVector sum = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      for (int column = 0; column < vectorBound; column += VALUES_PER_VECTOR) {
        FloatVector first =
            FloatVector.fromArray(
                PanamaVectorUtilSupport.FLOAT_SPECIES, input, inputOffset + column);
        FloatVector second =
            FloatVector.fromArray(
                PanamaVectorUtilSupport.FLOAT_SPECIES,
                input,
                inputOffset + column + PanamaVectorUtilSupport.FLOAT_SPECIES.length());
        sum =
            accumulate(
                weight,
                rowOffset + (long) column * BYTES_PER_VALUE,
                first.rearrange(EVEN_INPUTS, second),
                first.rearrange(ODD_INPUTS, second),
                sum);
      }
      output[outputOffset + row] =
          finishRow(weight, rowOffset, input, inputOffset, columns, vectorBound, sum);
    }
  }

  private void multiplyVectorBatch(
      float[] input, int inputOffset, int batchSize, float[] output, int outputOffset) {
    GgufParallelSupport.forEachRow(
        data,
        rows,
        columns,
        row -> multiplyVectorBatchRow(input, inputOffset, batchSize, output, outputOffset, row));
  }

  private void multiplyVectorBatchRow(
      float[] input, int inputOffset, int batchSize, float[] output, int outputOffset, int row) {
    int vectorBound = columns - columns % VALUES_PER_VECTOR;
    long rowOffset = (long) row * columns * BYTES_PER_VALUE;
    int batchGroup = batchSize & ~3;
    for (int batch = 0; batch < batchGroup; batch += 4) {
      int activation0 = inputOffset + batch * columns;
      int activation1 = activation0 + columns;
      int activation2 = activation1 + columns;
      int activation3 = activation2 + columns;
      FloatVector sum0 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      FloatVector sum1 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      FloatVector sum2 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      FloatVector sum3 = FloatVector.zero(PanamaVectorUtilSupport.FLOAT_SPECIES);
      for (int column = 0; column < vectorBound; column += VALUES_PER_VECTOR) {
        long byteOffset = rowOffset + (long) column * BYTES_PER_VALUE;
        DecodedWeights weights = decodeWeights(data, byteOffset);
        sum0 = accumulate(input, activation0 + column, weights, sum0);
        sum1 = accumulate(input, activation1 + column, weights, sum1);
        sum2 = accumulate(input, activation2 + column, weights, sum2);
        sum3 = accumulate(input, activation3 + column, weights, sum3);
      }
      output[outputOffset + batch * rows + row] =
          finishRow(data, rowOffset, input, activation0, columns, vectorBound, sum0);
      output[outputOffset + (batch + 1) * rows + row] =
          finishRow(data, rowOffset, input, activation1, columns, vectorBound, sum1);
      output[outputOffset + (batch + 2) * rows + row] =
          finishRow(data, rowOffset, input, activation2, columns, vectorBound, sum2);
      output[outputOffset + (batch + 3) * rows + row] =
          finishRow(data, rowOffset, input, activation3, columns, vectorBound, sum3);
    }
    for (int batch = batchGroup; batch < batchSize; batch++) {
      multiplyVectorRows(
          input,
          inputOffset + batch * columns,
          data.asSlice(rowOffset, (long) columns * BYTES_PER_VALUE),
          1,
          columns,
          output,
          outputOffset + batch * rows + row);
    }
  }

  private static FloatVector accumulate(
      float[] input, int inputOffset, DecodedWeights weights, FloatVector sum) {
    FloatVector first =
        FloatVector.fromArray(PanamaVectorUtilSupport.FLOAT_SPECIES, input, inputOffset);
    FloatVector second =
        FloatVector.fromArray(
            PanamaVectorUtilSupport.FLOAT_SPECIES,
            input,
            inputOffset + PanamaVectorUtilSupport.FLOAT_SPECIES.length());
    sum = PanamaVectorUtilSupport.fma(weights.even(), first.rearrange(EVEN_INPUTS, second), sum);
    return PanamaVectorUtilSupport.fma(weights.odd(), first.rearrange(ODD_INPUTS, second), sum);
  }

  private static FloatVector accumulate(
      MemorySegment weight,
      long byteOffset,
      FloatVector evenInputs,
      FloatVector oddInputs,
      FloatVector sum) {
    DecodedWeights weights = decodeWeights(weight, byteOffset);
    sum = PanamaVectorUtilSupport.fma(weights.even(), evenInputs, sum);
    return PanamaVectorUtilSupport.fma(weights.odd(), oddInputs, sum);
  }

  private static DecodedWeights decodeWeights(MemorySegment weight, long byteOffset) {
    IntVector packed =
        IntVector.fromMemorySegment(
            PanamaVectorUtilSupport.INT_SPECIES, weight, byteOffset, ByteOrder.LITTLE_ENDIAN);
    return new DecodedWeights(
        decodeHalves(packed.and(0xffff)),
        decodeHalves(packed.lanewise(VectorOperators.LSHR, Short.SIZE)));
  }

  private static FloatVector decodeHalves(IntVector halves) {
    IntVector absolute = halves.and(0x7fff);
    IntVector exponent = absolute.and(0x7c00);
    IntVector mantissa = absolute.and(0x03ff);
    IntVector sign = halves.and(0x8000).lanewise(VectorOperators.LSHL, Short.SIZE);
    IntVector bits = absolute.lanewise(VectorOperators.LSHL, 13).add(0x38000000).or(sign);
    bits = bits.blend(sign, absolute.compare(VectorOperators.EQ, 0));
    IntVector special = mantissa.lanewise(VectorOperators.LSHL, 13).or(0x7f800000).or(sign);
    bits = bits.blend(special, exponent.compare(VectorOperators.EQ, 0x7c00));
    FloatVector decoded = bits.reinterpretAsFloats();

    VectorMask<Integer> subnormal =
        exponent.compare(VectorOperators.EQ, 0).and(absolute.compare(VectorOperators.NE, 0));
    if (subnormal.anyTrue()) {
      FloatVector values = ((FloatVector) mantissa.convert(VectorOperators.I2F, 0)).mul(0x1.0p-24f);
      IntVector subnormalBits = values.reinterpretAsInts().or(sign);
      decoded =
          decoded.blend(
              subnormalBits.reinterpretAsFloats(),
              subnormal.cast(PanamaVectorUtilSupport.FLOAT_SPECIES));
    }
    return decoded;
  }

  private static float finishRow(
      MemorySegment weight,
      long rowOffset,
      float[] input,
      int inputOffset,
      int columns,
      int vectorBound,
      FloatVector vectorSum) {
    float sum = vectorSum.reduceLanes(VectorOperators.ADD);
    for (int column = vectorBound; column < columns; column++) {
      short bits = weight.get(LE_SHORT, rowOffset + (long) column * BYTES_PER_VALUE);
      sum = MathUtil.fma(input[inputOffset + column], Float.float16ToFloat(bits), sum);
    }
    return sum;
  }

  private float valueAt(long index) {
    return Float.float16ToFloat(data.get(LE_SHORT, index * BYTES_PER_VALUE));
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive; got " + value);
    }
  }

  private static void requireArrayRange(int length, int offset, int count, String name) {
    if (offset < 0 || count < 0 || offset > length - count) {
      throw new IllegalArgumentException(
          name + " range [" + offset + ", " + ((long) offset + count) + ") exceeds " + length);
    }
  }

  private record DecodedWeights(FloatVector even, FloatVector odd) {}
}
