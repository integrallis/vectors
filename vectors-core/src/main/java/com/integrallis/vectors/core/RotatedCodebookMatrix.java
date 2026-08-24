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
import java.util.Arrays;
import java.util.Objects;

/**
 * A matrix whose rows use grouped, Hadamard-rotated codebook quantization.
 *
 * <p>The packed codes and FP16 norms are supplied as separate memory segments, so callers may use
 * heap arrays or zero-copy slices of a mapped model file. Each group is reconstructed as {@code
 * (codebook[codes] * norm) * H}, where {@code H} is the normalized Walsh-Hadamard matrix.
 *
 * <p>Preparing an activation transforms each input group once. The result may be reused across
 * compatible matrices, such as query, key, and value projections with the same input geometry and
 * codebook.
 */
public final class RotatedCodebookMatrix {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final int LOOKUP_VALUES = 256;

  private final MemorySegment packedCodes;
  private final MemorySegment norms;
  private final int rows;
  private final int columns;
  private final int groupSize;
  private final int paddedColumns;
  private final int groupsPerRow;
  private final int rowBytes;
  private final Encoding encoding;
  private final float[] codebook;

  private RotatedCodebookMatrix(
      MemorySegment packedCodes,
      MemorySegment norms,
      int rows,
      int columns,
      int groupSize,
      int paddedColumns,
      int groupsPerRow,
      int rowBytes,
      Encoding encoding,
      float[] codebook) {
    this.packedCodes = packedCodes;
    this.norms = norms;
    this.rows = rows;
    this.columns = columns;
    this.groupSize = groupSize;
    this.paddedColumns = paddedColumns;
    this.groupsPerRow = groupsPerRow;
    this.rowBytes = rowBytes;
    this.encoding = encoding;
    this.codebook = codebook;
  }

  /**
   * Creates a matrix over packed code and little-endian FP16 norm storage.
   *
   * @param packedCodes row-major packed code indices
   * @param norms row-major group L2 norms as little-endian FP16 values
   * @param rows logical output rows
   * @param columns logical input columns
   * @param groupSize quantization group size; must be a power of two and at least eight
   * @param encoding packed-code encoding
   * @param codebook reconstruction values, already normalized for the group geometry
   * @return a validated matrix view
   */
  public static RotatedCodebookMatrix of(
      MemorySegment packedCodes,
      MemorySegment norms,
      int rows,
      int columns,
      int groupSize,
      Encoding encoding,
      float[] codebook) {
    Objects.requireNonNull(packedCodes, "packedCodes");
    Objects.requireNonNull(norms, "norms");
    Objects.requireNonNull(encoding, "encoding");
    Objects.requireNonNull(codebook, "codebook");
    requirePositive(rows, "rows");
    requirePositive(columns, "columns");
    if (groupSize < 8 || Integer.bitCount(groupSize) != 1) {
      throw new IllegalArgumentException("groupSize must be a power of two and at least 8");
    }
    if (codebook.length != encoding.codebookLength) {
      throw new IllegalArgumentException(
          "codebook length for "
              + encoding
              + " must be "
              + encoding.codebookLength
              + "; got "
              + codebook.length);
    }
    for (float value : codebook) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("codebook values must be finite");
      }
    }

    long groups = ((long) columns + groupSize - 1L) / groupSize;
    long padded = Math.multiplyExact(groups, groupSize);
    if (padded > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("padded columns exceed the supported Java array size");
    }
    long rowBytes = Math.multiplyExact(padded, encoding.packedBits) / Byte.SIZE;
    long requiredPackedBytes = Math.multiplyExact((long) rows, rowBytes);
    long requiredNormBytes =
        Math.multiplyExact(Math.multiplyExact((long) rows, groups), Short.BYTES);
    requireStorage(packedCodes, requiredPackedBytes, "packed codes");
    requireStorage(norms, requiredNormBytes, "norms");
    validateNorms(norms, rows, Math.toIntExact(groups));

    return new RotatedCodebookMatrix(
        packedCodes,
        norms,
        rows,
        columns,
        groupSize,
        Math.toIntExact(padded),
        Math.toIntExact(groups),
        Math.toIntExact(rowBytes),
        encoding,
        codebook.clone());
  }

  /** Returns the logical number of matrix rows. */
  public int rows() {
    return rows;
  }

  /** Returns the logical number of matrix columns. */
  public int columns() {
    return columns;
  }

  /** Returns the quantization group size. */
  public int groupSize() {
    return groupSize;
  }

  /** Returns the packed-code encoding. */
  public Encoding encoding() {
    return encoding;
  }

  /**
   * Transforms an input once for reuse across compatible matrices.
   *
   * @param input one logical input vector
   * @return a prepared activation owned by the caller
   */
  public PreparedActivation prepare(float[] input) {
    Objects.requireNonNull(input, "input");
    if (input.length != columns) {
      throw new IllegalArgumentException(
          "input length must equal matrix columns " + columns + "; got " + input.length);
    }
    for (float value : input) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("input values must be finite");
      }
    }

    float[] transformed = Arrays.copyOf(input, paddedColumns);
    float scale = (float) (1.0 / Math.sqrt(groupSize));
    for (int group = 0; group < groupsPerRow; group++) {
      int offset = group * groupSize;
      hadamardInPlace(transformed, offset, groupSize);
      for (int index = offset; index < offset + groupSize; index++) {
        transformed[index] *= scale;
      }
    }

    float[] byteLookup = encoding.usesByteLookup ? buildByteLookup(transformed) : null;
    return new PreparedActivation(
        columns, groupSize, rowBytes, encoding, codebook, transformed, byteLookup);
  }

  /**
   * Multiplies this matrix by a prepared activation.
   *
   * @param activation an activation prepared by a compatible matrix
   * @param output destination with room for every matrix row
   */
  public void multiply(PreparedActivation activation, float[] output) {
    Objects.requireNonNull(activation, "activation");
    Objects.requireNonNull(output, "output");
    if (!activation.compatible(columns, groupSize, rowBytes, encoding, codebook)) {
      throw new IllegalArgumentException("prepared activation is incompatible with this matrix");
    }
    if (output.length < rows) {
      throw new IllegalArgumentException(
          "output length must be at least matrix rows " + rows + "; got " + output.length);
    }

    if (encoding.usesByteLookup) {
      multiplyWithByteLookup(activation, output);
    } else {
      multiplyDirect(activation, output);
    }
  }

  private void multiplyWithByteLookup(PreparedActivation activation, float[] output) {
    int bytesPerGroup = rowBytes / groupsPerRow;
    for (int row = 0; row < rows; row++) {
      float sum = 0.0f;
      long rowOffset = (long) row * rowBytes;
      int normBase = row * groupsPerRow;
      for (int group = 0; group < groupsPerRow; group++) {
        float inner = 0.0f;
        int byteBase = group * bytesPerGroup;
        for (int index = 0; index < bytesPerGroup; index++) {
          int bytePosition = byteBase + index;
          int packed =
              Byte.toUnsignedInt(packedCodes.get(ValueLayout.JAVA_BYTE, rowOffset + bytePosition));
          float contribution = activation.byteLookup[bytePosition * LOOKUP_VALUES + packed];
          if (Float.isNaN(contribution)) {
            throw new IllegalArgumentException("packed codes contain invalid ternary crumb 2");
          }
          inner += contribution;
        }
        sum += norm(normBase + group) * inner;
      }
      output[row] = sum;
    }
  }

  private void multiplyDirect(PreparedActivation activation, float[] output) {
    for (int row = 0; row < rows; row++) {
      float sum = 0.0f;
      long rowOffset = (long) row * rowBytes;
      int normBase = row * groupsPerRow;
      for (int group = 0; group < groupsPerRow; group++) {
        float inner = 0.0f;
        int valueBase = group * groupSize;
        for (int index = 0; index < groupSize; index++) {
          int code = extractCode(rowOffset, valueBase + index);
          inner += codebook[code] * activation.transformed[valueBase + index];
        }
        sum += norm(normBase + group) * inner;
      }
      output[row] = sum;
    }
  }

  private float[] buildByteLookup(float[] transformed) {
    float[] table = new float[Math.multiplyExact(rowBytes, LOOKUP_VALUES)];
    int valuesPerByte = Byte.SIZE / encoding.packedBits;
    int mask = (1 << encoding.packedBits) - 1;
    for (int bytePosition = 0; bytePosition < rowBytes; bytePosition++) {
      int valueBase = bytePosition * valuesPerByte;
      int tableBase = bytePosition * LOOKUP_VALUES;
      for (int packed = 0; packed < LOOKUP_VALUES; packed++) {
        float contribution = 0.0f;
        boolean valid = true;
        for (int value = 0; value < valuesPerByte; value++) {
          int code = (packed >>> (value * encoding.packedBits)) & mask;
          int codebookIndex = encoding.codebookIndex(code);
          if (codebookIndex < 0) {
            valid = false;
            break;
          }
          contribution += codebook[codebookIndex] * transformed[valueBase + value];
        }
        table[tableBase + packed] = valid ? contribution : Float.NaN;
      }
    }
    return table;
  }

  private int extractCode(long rowOffset, int valueIndex) {
    int bitIndex = valueIndex * encoding.packedBits;
    long byteIndex = rowOffset + (bitIndex >>> 3);
    int shift = bitIndex & 7;
    int code = Byte.toUnsignedInt(packedCodes.get(ValueLayout.JAVA_BYTE, byteIndex)) >>> shift;
    int availableBits = Byte.SIZE - shift;
    if (availableBits < encoding.packedBits) {
      code |=
          Byte.toUnsignedInt(packedCodes.get(ValueLayout.JAVA_BYTE, byteIndex + 1L))
              << availableBits;
    }
    return code & ((1 << encoding.packedBits) - 1);
  }

  private float norm(int index) {
    return Float.float16ToFloat(norms.get(LE_SHORT, (long) index * Short.BYTES));
  }

  private static void hadamardInPlace(float[] values, int offset, int length) {
    for (int stride = 1; stride < length; stride *= 2) {
      for (int base = offset; base < offset + length; base += 2 * stride) {
        for (int index = 0; index < stride; index++) {
          float left = values[base + index];
          float right = values[base + stride + index];
          values[base + index] = left + right;
          values[base + stride + index] = left - right;
        }
      }
    }
  }

  private static void validateNorms(MemorySegment norms, int rows, int groupsPerRow) {
    int count = Math.multiplyExact(rows, groupsPerRow);
    for (int index = 0; index < count; index++) {
      float norm = Float.float16ToFloat(norms.get(LE_SHORT, (long) index * Short.BYTES));
      if (!Float.isFinite(norm) || norm < 0.0f) {
        throw new IllegalArgumentException("norms must contain finite, non-negative FP16 values");
      }
    }
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive");
    }
  }

  private static void requireStorage(MemorySegment segment, long required, String name) {
    if (segment.byteSize() < required) {
      throw new IllegalArgumentException(
          name + " require " + required + " bytes; got " + segment.byteSize());
    }
  }

  /** Packing conventions supported by the rotated-codebook primitive. */
  public enum Encoding {
    /** Four codebook indices packed into each byte. */
    CQ2(2, 4, true),

    /** Eight indices packed into each three-byte little-endian word. */
    CQ3(3, 8, false),

    /** Two codebook indices packed into each byte. */
    CQ4(4, 16, true),

    /** Signed two-bit crumbs {@code 3, 0, 1} represent codebook entries {@code 0, 1, 2}. */
    TERNARY(2, 3, true);

    private final int packedBits;
    private final int codebookLength;
    private final boolean usesByteLookup;

    Encoding(int packedBits, int codebookLength, boolean usesByteLookup) {
      this.packedBits = packedBits;
      this.codebookLength = codebookLength;
      this.usesByteLookup = usesByteLookup;
    }

    private int codebookIndex(int code) {
      if (this != TERNARY) {
        return code;
      }
      return switch (code) {
        case 3 -> 0;
        case 0 -> 1;
        case 1 -> 2;
        default -> -1;
      };
    }
  }

  /** A transformed activation that can be reused by compatible matrices. */
  public static final class PreparedActivation {
    private final int columns;
    private final int groupSize;
    private final int rowBytes;
    private final Encoding encoding;
    private final float[] codebook;
    private final float[] transformed;
    private final float[] byteLookup;

    private PreparedActivation(
        int columns,
        int groupSize,
        int rowBytes,
        Encoding encoding,
        float[] codebook,
        float[] transformed,
        float[] byteLookup) {
      this.columns = columns;
      this.groupSize = groupSize;
      this.rowBytes = rowBytes;
      this.encoding = encoding;
      this.codebook = codebook;
      this.transformed = transformed;
      this.byteLookup = byteLookup;
    }

    private boolean compatible(
        int columns, int groupSize, int rowBytes, Encoding encoding, float[] codebook) {
      return this.columns == columns
          && this.groupSize == groupSize
          && this.rowBytes == rowBytes
          && this.encoding == encoding
          && (this.codebook == codebook || Arrays.equals(this.codebook, codebook));
    }
  }
}
