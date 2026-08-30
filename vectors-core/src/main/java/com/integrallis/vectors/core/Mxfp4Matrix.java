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
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;

/** A row-major MXFP4 matrix with E2M1 values and one E8M0 scale per 32 columns. */
public final class Mxfp4Matrix {

  private static final int VALUES_PER_BLOCK = 32;
  private static final int PACKED_BYTES_PER_BLOCK = VALUES_PER_BLOCK / 2;
  private static final boolean USE_VECTOR_256 =
      VectorizationProvider.runtimeCapabilities().activeVectorBits() >= 256;
  private static final byte[] E2M1_X2 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  private static final ByteVector E2M1_X2_LUT =
      ByteVector.fromArray(ByteVector.SPECIES_128, E2M1_X2, 0);
  private static final VectorShuffle<Byte> EVEN_ACTIVATIONS =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_128, 0);
  private static final VectorShuffle<Byte> ODD_ACTIVATIONS =
      VectorShuffle.makeUnzip(ByteVector.SPECIES_128, 1);
  private static final float[] E2M1 = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
  };

  private final MemorySegment blocks;
  private final MemorySegment scales;
  private final int rows;
  private final int columns;
  private final int blocksPerRow;
  private final int packedBytesPerRow;

  private Mxfp4Matrix(
      MemorySegment blocks,
      MemorySegment scales,
      int rows,
      int columns,
      int blocksPerRow,
      int packedBytesPerRow) {
    this.blocks = blocks.asReadOnly();
    this.scales = scales.asReadOnly();
    this.rows = rows;
    this.columns = columns;
    this.blocksPerRow = blocksPerRow;
    this.packedBytesPerRow = packedBytesPerRow;
  }

  /**
   * Creates a zero-copy view over the standard MXFP4 row layout.
   *
   * @param blocks packed E2M1 values; each byte stores the earlier value in its low nibble
   * @param scales unsigned E8M0 scale codes, one per 32 consecutive input columns
   * @param rows logical output rows
   * @param columns logical input columns, which must be a multiple of 32
   * @return a validated read-only matrix view
   */
  public static Mxfp4Matrix of(MemorySegment blocks, MemorySegment scales, int rows, int columns) {
    Objects.requireNonNull(blocks, "blocks");
    Objects.requireNonNull(scales, "scales");
    requirePositive(rows, "rows");
    requirePositive(columns, "columns");
    if (columns % VALUES_PER_BLOCK != 0) {
      throw new IllegalArgumentException("columns must be a multiple of 32: " + columns);
    }
    int blocksPerRow = columns / VALUES_PER_BLOCK;
    int packedBytesPerRow = Math.multiplyExact(blocksPerRow, PACKED_BYTES_PER_BLOCK);
    long requiredBlockBytes = Math.multiplyExact((long) rows, packedBytesPerRow);
    long requiredScaleBytes = Math.multiplyExact((long) rows, blocksPerRow);
    requireStorage(blocks, requiredBlockBytes, "blocks");
    requireStorage(scales, requiredScaleBytes, "scales");
    return new Mxfp4Matrix(blocks, scales, rows, columns, blocksPerRow, packedBytesPerRow);
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
    checkMultiplyArguments(input, output);

    for (int row = 0; row < rows; row++) {
      float sum = 0.0f;
      long rowBlockOffset = (long) row * packedBytesPerRow;
      long rowScaleOffset = (long) row * blocksPerRow;
      for (int block = 0; block < blocksPerRow; block++) {
        float scale =
            decodeScale(
                Byte.toUnsignedInt(scales.get(ValueLayout.JAVA_BYTE, rowScaleOffset + block)));
        long packedOffset = rowBlockOffset + (long) block * PACKED_BYTES_PER_BLOCK;
        int inputOffset = block * VALUES_PER_BLOCK;
        for (int packed = 0; packed < PACKED_BYTES_PER_BLOCK; packed++) {
          int codes = Byte.toUnsignedInt(blocks.get(ValueLayout.JAVA_BYTE, packedOffset + packed));
          sum = MathUtil.fma(E2M1[codes & 0x0f] * scale, input[inputOffset + 2 * packed], sum);
          sum = MathUtil.fma(E2M1[codes >>> 4] * scale, input[inputOffset + 2 * packed + 1], sum);
        }
      }
      output[row] = sum;
    }
  }

  /**
   * Multiplies using caller-owned Q8_0 activation storage. This W4A8 path trades a small activation
   * quantization error for integer dot-product throughput while retaining the original MXFP4
   * weights.
   */
  public void multiplyQ8(float[] input, float[] output, GgufQ8_0Batch activation) {
    checkMultiplyArguments(input, output);
    checkActivation(activation);
    activation.quantize(input, 1);
    multiplyQ8(activation, output);
  }

  /**
   * Multiplies using the first row of an already-quantized Q8_0 activation. Preparing the
   * activation once and passing it to several matrices avoids repeating quantization for
   * projections that share the same input.
   */
  public void multiplyQ8(GgufQ8_0Batch activation, float[] output) {
    checkActivation(activation);
    checkOutput(output);
    byte[] quants = activation.quants();
    float[] activationScales = activation.scales();

    GgufParallelSupport.forEachRow(
        blocks, rows, columns, row -> output[row] = q8RowDot(row, quants, activationScales));
  }

  private float q8RowDot(int row, byte[] quants, float[] activationScales) {
    float sum = 0.0f;
    long rowOffset = (long) row * packedBytesPerRow;
    for (int block = 0; block < blocksPerRow; block++) {
      int scaleCode =
          Byte.toUnsignedInt(scales.get(ValueLayout.JAVA_BYTE, (long) row * blocksPerRow + block));
      float combinedScale = 0.5f * decodeScale(scaleCode) * activationScales[block];
      int integerSum =
          integerDot(
              blocks,
              rowOffset + (long) block * PACKED_BYTES_PER_BLOCK,
              quants,
              block * VALUES_PER_BLOCK);
      sum = MathUtil.fma(combinedScale, integerSum, sum);
    }
    return sum;
  }

  private static int integerDot(
      MemorySegment weights, long weightOffset, byte[] quants, int quantOffset) {
    if (!USE_VECTOR_256) {
      int sum = 0;
      for (int pair = 0; pair < PACKED_BYTES_PER_BLOCK; pair++) {
        int codes = Byte.toUnsignedInt(weights.get(ValueLayout.JAVA_BYTE, weightOffset + pair));
        sum += E2M1_X2[codes & 0x0f] * quants[quantOffset + 2 * pair];
        sum += E2M1_X2[codes >>> 4] * quants[quantOffset + 2 * pair + 1];
      }
      return sum;
    }

    ByteVector packed =
        ByteVector.fromMemorySegment(
            ByteVector.SPECIES_128, weights, weightOffset, ByteOrder.LITTLE_ENDIAN);
    ByteVector lowWeights = packed.and((byte) 0x0f).selectFrom(E2M1_X2_LUT);
    ByteVector highWeights = packed.lanewise(VectorOperators.LSHR, 4).selectFrom(E2M1_X2_LUT);
    ByteVector first = ByteVector.fromArray(ByteVector.SPECIES_128, quants, quantOffset);
    ByteVector second = ByteVector.fromArray(ByteVector.SPECIES_128, quants, quantOffset + 16);
    ByteVector evenQuants = first.rearrange(EVEN_ACTIVATIONS, second);
    ByteVector oddQuants = first.rearrange(ODD_ACTIVATIONS, second);
    ShortVector lowProducts =
        ((ShortVector) lowWeights.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0))
            .mul(
                (ShortVector)
                    evenQuants.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0));
    ShortVector highProducts =
        ((ShortVector) highWeights.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0))
            .mul(
                (ShortVector)
                    oddQuants.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0));
    return lowProducts.reduceLanes(VectorOperators.ADD)
        + highProducts.reduceLanes(VectorOperators.ADD);
  }

  /** Reads one matrix value as F32. */
  public float value(int row, int column) {
    if (row < 0 || row >= rows || column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException(
          "matrix coordinate (" + row + ", " + column + ") is outside " + rows + "x" + columns);
    }
    int block = column / VALUES_PER_BLOCK;
    int withinBlock = column % VALUES_PER_BLOCK;
    long packedOffset =
        (long) row * packedBytesPerRow + (long) block * PACKED_BYTES_PER_BLOCK + withinBlock / 2;
    int codes = Byte.toUnsignedInt(blocks.get(ValueLayout.JAVA_BYTE, packedOffset));
    int code = withinBlock % 2 == 0 ? codes & 0x0f : codes >>> 4;
    int scaleCode =
        Byte.toUnsignedInt(scales.get(ValueLayout.JAVA_BYTE, (long) row * blocksPerRow + block));
    return E2M1[code] * decodeScale(scaleCode);
  }

  /** Returns a zero-copy view over a contiguous range of rows. */
  public Mxfp4Matrix rowSlice(int fromRow, int rowCount) {
    if (fromRow < 0 || rowCount <= 0 || fromRow > rows - rowCount) {
      throw new IndexOutOfBoundsException(
          "row range [" + fromRow + ", " + ((long) fromRow + rowCount) + ") exceeds " + rows);
    }
    long blockOffset = Math.multiplyExact((long) fromRow, packedBytesPerRow);
    long blockBytes = Math.multiplyExact((long) rowCount, packedBytesPerRow);
    long scaleOffset = Math.multiplyExact((long) fromRow, blocksPerRow);
    long scaleBytes = Math.multiplyExact((long) rowCount, blocksPerRow);
    return new Mxfp4Matrix(
        blocks.asSlice(blockOffset, blockBytes),
        scales.asSlice(scaleOffset, scaleBytes),
        rowCount,
        columns,
        blocksPerRow,
        packedBytesPerRow);
  }

  private static float decodeScale(int code) {
    return Math.scalb(1.0f, Math.min(code, 254) - 127);
  }

  private void checkMultiplyArguments(float[] input, float[] output) {
    Objects.requireNonNull(input, "input");
    if (input.length != columns) {
      throw new IllegalArgumentException(
          "input length must equal matrix columns " + columns + "; got " + input.length);
    }
    checkOutput(output);
  }

  private void checkOutput(float[] output) {
    Objects.requireNonNull(output, "output");
    if (output.length < rows) {
      throw new IllegalArgumentException(
          "output length must be at least matrix rows " + rows + "; got " + output.length);
    }
  }

  private void checkActivation(GgufQ8_0Batch activation) {
    Objects.requireNonNull(activation, "activation");
    if (activation.dimensions() != columns || activation.batchCapacity() < 1) {
      throw new IllegalArgumentException(
          "activation storage must retain at least one row of " + columns + " dimensions");
    }
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive: " + value);
    }
  }

  private static void requireStorage(MemorySegment storage, long required, String name) {
    if (storage.byteSize() != required) {
      throw new IllegalArgumentException(
          name + " requires " + required + " bytes; got " + storage.byteSize());
    }
  }
}
