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

/**
 * Interface for all SIMD-accelerable vector operations. Implementations are selected at runtime by
 * {@link VectorizationProvider}: Panama Vector API (preferred) or scalar fallback.
 *
 * <p>Modeled after Apache Lucene's {@code VectorUtilSupport} and JVector's {@code
 * VectorUtilSupport}, combining the best patterns from both.
 */
public interface VectorUtilSupport {

  ValueLayout.OfShort GGUF_LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  ValueLayout.OfInt GGUF_LE_INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  ValueLayout.OfFloat GGUF_LE_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  int GGUF_Q_BLOCK_SIZE = 32;
  int GGUF_Q4_0_BLOCK_BYTES = 18;
  int GGUF_Q5_0_BLOCK_BYTES = 22;
  int GGUF_Q8_0_BLOCK_BYTES = 34;
  int GGUF_Q4_K_BLOCK_SIZE = 256;
  int GGUF_Q4_K_BLOCK_BYTES = 144;
  int GGUF_Q4_K_SCALES_OFFSET = 4;
  int GGUF_Q4_K_QUANTS_OFFSET = 16;
  int GGUF_Q5_K_BLOCK_SIZE = 256;
  int GGUF_Q5_K_BLOCK_BYTES = 176;
  int GGUF_Q5_K_SCALES_OFFSET = 4;
  int GGUF_Q5_K_HIGH_BITS_OFFSET = 16;
  int GGUF_Q5_K_QUANTS_OFFSET = 48;
  int GGUF_Q8_K_SUM_BLOCK_SIZE = 16;
  int GGUF_Q6_K_BLOCK_SIZE = 256;
  int GGUF_Q6_K_BLOCK_BYTES = 210;
  int GGUF_Q6_K_QL_BYTES = 128;
  int GGUF_Q6_K_QH_BYTES = 64;
  int GGUF_Q6_K_SCALES = 16;

  // --- Float distance kernels ---

  /** Computes the dot product of two float vectors. */
  float dotProduct(float[] a, float[] b);

  /** Computes the dot product of sub-vectors at the given offsets. */
  float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length);

  /** Computes the squared Euclidean (L2) distance between two float vectors. */
  float squareDistance(float[] a, float[] b);

  /** Computes the squared Euclidean distance of sub-vectors at the given offsets. */
  float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length);

  /** Computes the cosine similarity between two float vectors. Returns value in [-1, 1]. */
  float cosine(float[] a, float[] b);

  // --- Byte distance kernels ---

  /** Computes the dot product of two signed byte vectors. */
  int dotProduct(byte[] a, byte[] b);

  /** Computes the squared Euclidean distance between two signed byte vectors. */
  int squareDistance(byte[] a, byte[] b);

  /** Computes the cosine similarity between two signed byte vectors. */
  float cosine(byte[] a, byte[] b);

  // --- MemorySegment distance kernels (off-heap) ---

  /** Computes the dot product from off-heap float vectors stored in MemorySegments. */
  float dotProduct(MemorySegment a, MemorySegment b, int dimensions);

  /** Computes the squared L2 distance from off-heap float vectors stored in MemorySegments. */
  float squareDistance(MemorySegment a, MemorySegment b, int dimensions);

  /**
   * Computes the cosine similarity from off-heap float vectors stored in MemorySegments. Returns
   * value in [-1, 1].
   */
  float cosine(MemorySegment a, MemorySegment b, int dimensions);

  // --- Binary distance ---

  /** Computes the Hamming distance between two binary vectors stored as packed long arrays. */
  int hammingDistance(long[] a, long[] b);

  // --- Vector arithmetic ---

  /** Adds v2 to v1 element-wise, storing the result in v1. */
  void addInPlace(float[] v1, float[] v2);

  /** Adds {@code vector * scale} to {@code out} over the requested sub-vector range. */
  default void addScaledInPlace(
      float[] out, int outOffset, float[] vector, int vectorOffset, int length, float scale) {
    for (int i = 0; i < length; i++) {
      int outIndex = outOffset + i;
      out[outIndex] = MathUtil.fma(vector[vectorOffset + i], scale, out[outIndex]);
    }
  }

  /** Subtracts v2 from v1 element-wise, storing the result in v1. */
  void subInPlace(float[] v1, float[] v2);

  /** Scales each element of the vector by the given multiplier. */
  void scale(float[] vector, float multiplier);

  /** Returns the sum of all elements in the vector. */
  float sum(float[] vector);

  // --- Normalization ---

  /**
   * L2-normalizes the given vector in place and returns it.
   *
   * @param v the vector to normalize
   * @param throwOnZero if true, throws IllegalArgumentException for zero-length vectors
   * @return the normalized vector (same array reference)
   */
  float[] l2normalize(float[] v, boolean throwOnZero);

  // --- Fused batch matrix-vector kernels (GEMV) ---

  /**
   * Fused matrix-vector dot product: fills {@code out[i] = dot(query, matrix[i])} for {@code i in
   * [0, numRows)}.
   *
   * <p>The default implementation is a simple loop. SIMD subclasses override this to load each
   * query SIMD chunk <em>once</em> and apply it to 4 matrix rows simultaneously, cutting query
   * memory traffic by 4× compared to calling {@link #dotProduct} per row.
   *
   * @param query the query vector (length = {@code matrix[0].length})
   * @param matrix the matrix rows (each row must have the same length as {@code query})
   * @param out the output array (must have length &ge; {@code numRows})
   * @param numRows the number of rows to process (must be &le; {@code matrix.length})
   */
  default void matVecDot(float[] query, float[][] matrix, float[] out, int numRows) {
    for (int i = 0; i < numRows; i++) {
      out[i] = dotProduct(query, 0, matrix[i], 0, query.length);
    }
  }

  /**
   * Fused matrix-vector dot product over a flat row-major matrix: fills {@code out[row] =
   * dot(query, matrix[row])} for {@code row in [0, rows)}.
   *
   * <p>This is the allocation-free shape used by dense model tensors and mmap-backed loaders that
   * expose row-major tensor payloads as a single contiguous buffer. SIMD subclasses override this
   * with the same 4-row query-load amortization used by the {@code float[][]} batch kernel.
   *
   * @param query the query vector (length = {@code cols})
   * @param rowMajorMatrix flat row-major matrix of length at least {@code rows * cols}
   * @param rows the number of matrix rows to process
   * @param cols the number of float elements per row
   * @param out the output array (must have length &ge; {@code rows})
   */
  default void matVecDot(float[] query, float[] rowMajorMatrix, int rows, int cols, float[] out) {
    for (int row = 0; row < rows; row++) {
      out[row] = dotProduct(query, 0, rowMajorMatrix, row * cols, cols);
    }
  }

  /** Batched row-major GEMV over little-endian GGUF F32 rows without copying mapped weights. */
  default void ggufF32MatVecDot(
      float[] query, MemorySegment weight, int rows, int cols, float[] out) {
    long rowBytes = (long) cols * Float.BYTES;
    for (int row = 0; row < rows; row++) {
      long rowOffset = row * rowBytes;
      float sum = 0.0f;
      for (int col = 0; col < cols; col++) {
        float value = weight.get(GGUF_LE_FLOAT, rowOffset + (long) col * Float.BYTES);
        sum = MathUtil.fma(query[col], value, sum);
      }
      out[row] = sum;
    }
  }

  /**
   * Dot product of a full-precision query with one GGUF Q4_0 quantized row.
   *
   * <p>Q4_0 stores each 32-value block as a little-endian binary16 scale followed by 16 packed
   * unsigned nibbles. Each nibble is centered by subtracting 8, then multiplied by the block scale.
   * This kernel fuses dequantization and dot product without materializing a temporary float row.
   */
  default float ggufQ4_0DotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufBlockAligned(dimensions);
    float sum = 0f;
    int blocks = dimensions / GGUF_Q_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
      float scale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      long nibbleOffset = blockOffset + Short.BYTES;
      int queryOffset = block * GGUF_Q_BLOCK_SIZE;
      for (int i = 0; i < 16; i++) {
        int packed = qWeight.get(ValueLayout.JAVA_BYTE, nibbleOffset + i) & 0xFF;
        int lo = (packed & 0x0F) - 8;
        int hi = ((packed >>> 4) & 0x0F) - 8;
        sum = MathUtil.fma(query[queryOffset + i], lo * scale, sum);
        sum = MathUtil.fma(query[queryOffset + i + 16], hi * scale, sum);
      }
    }
    return sum;
  }

  /**
   * Dot product of a full-precision query with one GGUF Q4_K quantized row.
   *
   * <p>Each 256-value super-block stores binary16 scale and minimum factors, eight packed 6-bit
   * scale/minimum pairs, and eight groups of 32 unsigned 4-bit quants. This follows GGML's Q4_K
   * layout and fuses dequantization with accumulation.
   */
  default float ggufQ4_KDotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQ4_KBlockAligned(dimensions);
    float sum = 0.0f;
    int blocks = dimensions / GGUF_Q4_K_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      float dMin = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
      long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
      int queryOffset = block * GGUF_Q4_K_BLOCK_SIZE;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        float scaledD = d * scale;
        float minimum = dMin * min;
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int groupQueryOffset = queryOffset + group * 32;

        for (int index = 0; index < 32; index++) {
          int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
          int quant = (packed >>> shift) & 0x0F;
          float weight = scaledD * quant - minimum;
          sum = MathUtil.fma(query[groupQueryOffset + index], weight, sum);
        }
      }
    }
    return sum;
  }

  /** Dequantizes GGUF Q4_K blocks into a caller-owned float array. */
  default void ggufQ4_KDequantize(
      MemorySegment qWeight, long byteOffset, float[] out, int outOffset, int dimensions) {
    checkGgufQ4_KBlockAligned(dimensions);
    int blocks = dimensions / GGUF_Q4_K_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      float dMin = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
      long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
      int outputOffset = outOffset + block * GGUF_Q4_K_BLOCK_SIZE;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        float scaledD = d * scale;
        float minimum = dMin * min;
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int groupOutputOffset = outputOffset + group * 32;

        for (int index = 0; index < 32; index++) {
          int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
          int quant = (packed >>> shift) & 0x0F;
          out[groupOutputOffset + index] = scaledD * quant - minimum;
        }
      }
    }
  }

  /**
   * Dot product of a full-precision query with one GGUF Q5_K quantized row.
   *
   * <p>Each 256-value super-block stores binary16 scale and minimum factors, eight packed 6-bit
   * scale/minimum pairs, 32 high-bit bytes, and eight groups of 32 low nibbles.
   */
  default float ggufQ5_KDotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQ5_KBlockAligned(dimensions);
    float sum = 0.0f;
    int blocks = dimensions / GGUF_Q5_K_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q5_K_BLOCK_BYTES;
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      float dMin = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
      long scalesOffset = blockOffset + GGUF_Q5_K_SCALES_OFFSET;
      long highBitsOffset = blockOffset + GGUF_Q5_K_HIGH_BITS_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q5_K_QUANTS_OFFSET;
      int queryOffset = block * GGUF_Q5_K_BLOCK_SIZE;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        float scaledD = d * scale;
        float minimum = dMin * min;
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int highBit = 1 << group;
        int groupQueryOffset = queryOffset + group * 32;

        for (int index = 0; index < 32; index++) {
          int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
          int highBits = qWeight.get(ValueLayout.JAVA_BYTE, highBitsOffset + index) & 0xFF;
          int quant = ((packed >>> shift) & 0x0F) | ((highBits & highBit) == 0 ? 0 : 16);
          float weight = scaledD * quant - minimum;
          sum = MathUtil.fma(query[groupQueryOffset + index], weight, sum);
        }
      }
    }
    return sum;
  }

  /** Dequantizes GGUF Q5_K blocks into a caller-owned float array. */
  default void ggufQ5_KDequantize(
      MemorySegment qWeight, long byteOffset, float[] out, int outOffset, int dimensions) {
    checkGgufQ5_KBlockAligned(dimensions);
    int blocks = dimensions / GGUF_Q5_K_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q5_K_BLOCK_BYTES;
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      float dMin = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES));
      long scalesOffset = blockOffset + GGUF_Q5_K_SCALES_OFFSET;
      long highBitsOffset = blockOffset + GGUF_Q5_K_HIGH_BITS_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q5_K_QUANTS_OFFSET;
      int outputOffset = outOffset + block * GGUF_Q5_K_BLOCK_SIZE;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        float scaledD = d * scale;
        float minimum = dMin * min;
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int highBit = 1 << group;
        int groupOutputOffset = outputOffset + group * 32;

        for (int index = 0; index < 32; index++) {
          int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
          int highBits = qWeight.get(ValueLayout.JAVA_BYTE, highBitsOffset + index) & 0xFF;
          int quant = ((packed >>> shift) & 0x0F) | ((highBits & highBit) == 0 ? 0 : 16);
          out[groupOutputOffset + index] = scaledD * quant - minimum;
        }
      }
    }
  }

  /**
   * Dot product of a full-precision query with one GGUF Q5_0 quantized row.
   *
   * <p>Each 32-value block stores a binary16 scale, a 32-bit high-bit mask, and 16 split low
   * nibbles. The reconstructed unsigned 5-bit value is centered by subtracting 16.
   */
  default float ggufQ5_0DotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufBlockAligned(dimensions);
    float sum = 0.0f;
    int blocks = dimensions / GGUF_Q_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q5_0_BLOCK_BYTES;
      float scale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      int highBits = qWeight.get(GGUF_LE_INT, blockOffset + Short.BYTES);
      long quantsOffset = blockOffset + Short.BYTES + Integer.BYTES;
      int queryOffset = block * GGUF_Q_BLOCK_SIZE;
      for (int index = 0; index < 16; index++) {
        int packed = qWeight.get(ValueLayout.JAVA_BYTE, quantsOffset + index) & 0xFF;
        int low = (packed & 0x0F) | (((highBits >>> index) & 1) << 4);
        int high = (packed >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
        sum = MathUtil.fma(query[queryOffset + index], (low - 16) * scale, sum);
        sum = MathUtil.fma(query[queryOffset + index + 16], (high - 16) * scale, sum);
      }
    }
    return sum;
  }

  /**
   * Dot product of a full-precision query with one GGUF Q8_0 quantized row.
   *
   * <p>Q8_0 stores each 32-value block as a little-endian binary16 scale followed by 32 signed int8
   * values. This kernel fuses dequantization and dot product without materializing a temporary
   * float row.
   */
  default float ggufQ8_0DotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufBlockAligned(dimensions);
    float sum = 0f;
    int blocks = dimensions / GGUF_Q_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
      float scale = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset));
      long quantOffset = blockOffset + Short.BYTES;
      int queryOffset = block * GGUF_Q_BLOCK_SIZE;
      for (int i = 0; i < GGUF_Q_BLOCK_SIZE; i++) {
        byte quant = qWeight.get(ValueLayout.JAVA_BYTE, quantOffset + i);
        sum = MathUtil.fma(query[queryOffset + i], quant * scale, sum);
      }
    }
    return sum;
  }

  /**
   * Dot product of a full-precision query with one GGUF Q6_K quantized row.
   *
   * <p>Q6_K stores each 256-value super-block as lower 4-bit quants, upper 2-bit quants, sixteen
   * signed int8 sub-block scales, and one little-endian binary16 super-block scale. This kernel
   * follows the upstream GGML bit layout and fuses dequantization and dot product without
   * materializing a temporary float row.
   */
  default float ggufQ6_KDotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQ6_KBlockAligned(dimensions);
    float sum = 0f;
    int blocks = dimensions / GGUF_Q6_K_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q6_K_BLOCK_BYTES;
      float d =
          Float.float16ToFloat(
              qWeight.get(
                  GGUF_LE_SHORT,
                  blockOffset + GGUF_Q6_K_QL_BYTES + GGUF_Q6_K_QH_BYTES + GGUF_Q6_K_SCALES));
      long qlOffset = blockOffset;
      long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
      long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
      int queryOffset = block * GGUF_Q6_K_BLOCK_SIZE;

      for (int superBlock = 0; superBlock < 2; superBlock++) {
        long qlBase = qlOffset + (long) superBlock * 64;
        long qhBase = qhOffset + (long) superBlock * 32;
        long scaleBase = scaleOffset + (long) superBlock * 8;
        int outBase = queryOffset + superBlock * 128;

        for (int l = 0; l < 32; l++) {
          int is = l / 16;
          int ql1 = qWeight.get(ValueLayout.JAVA_BYTE, qlBase + l) & 0xFF;
          int ql2 = qWeight.get(ValueLayout.JAVA_BYTE, qlBase + 32L + l) & 0xFF;
          int qh = qWeight.get(ValueLayout.JAVA_BYTE, qhBase + l) & 0xFF;
          int q1 = ((ql1 & 0x0F) | ((qh & 0x03) << 4)) - 32;
          int q2 = ((ql2 & 0x0F) | (((qh >>> 2) & 0x03) << 4)) - 32;
          int q3 = ((ql1 >>> 4) | (((qh >>> 4) & 0x03) << 4)) - 32;
          int q4 = ((ql2 >>> 4) | (((qh >>> 6) & 0x03) << 4)) - 32;

          float d1 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + is);
          float d2 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + is + 2L);
          float d3 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + is + 4L);
          float d4 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + is + 6L);

          sum = MathUtil.fma(query[outBase + l], d1 * q1, sum);
          sum = MathUtil.fma(query[outBase + l + 32], d2 * q2, sum);
          sum = MathUtil.fma(query[outBase + l + 64], d3 * q3, sum);
          sum = MathUtil.fma(query[outBase + l + 96], d4 * q4, sum);
        }
      }
    }
    return sum;
  }

  /** Dequantizes GGUF Q6_K blocks into a caller-owned float array. */
  default void ggufQ6_KDequantize(
      MemorySegment qWeight, long byteOffset, float[] out, int outOffset, int dimensions) {
    checkGgufQ6_KBlockAligned(dimensions);
    int blocks = dimensions / GGUF_Q6_K_BLOCK_SIZE;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = byteOffset + (long) block * GGUF_Q6_K_BLOCK_BYTES;
      float d =
          Float.float16ToFloat(
              qWeight.get(
                  GGUF_LE_SHORT,
                  blockOffset + GGUF_Q6_K_QL_BYTES + GGUF_Q6_K_QH_BYTES + GGUF_Q6_K_SCALES));
      long qlOffset = blockOffset;
      long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
      long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
      int outputOffset = outOffset + block * GGUF_Q6_K_BLOCK_SIZE;

      for (int superBlock = 0; superBlock < 2; superBlock++) {
        long qlBase = qlOffset + (long) superBlock * 64;
        long qhBase = qhOffset + (long) superBlock * 32;
        long scaleBase = scaleOffset + (long) superBlock * 8;
        int outBase = outputOffset + superBlock * 128;

        for (int index = 0; index < 32; index++) {
          int scaleIndex = index / 16;
          int ql1 = qWeight.get(ValueLayout.JAVA_BYTE, qlBase + index) & 0xFF;
          int ql2 = qWeight.get(ValueLayout.JAVA_BYTE, qlBase + 32L + index) & 0xFF;
          int qh = qWeight.get(ValueLayout.JAVA_BYTE, qhBase + index) & 0xFF;
          int q1 = ((ql1 & 0x0F) | ((qh & 0x03) << 4)) - 32;
          int q2 = ((ql2 & 0x0F) | (((qh >>> 2) & 0x03) << 4)) - 32;
          int q3 = ((ql1 >>> 4) | (((qh >>> 4) & 0x03) << 4)) - 32;
          int q4 = ((ql2 >>> 4) | (((qh >>> 6) & 0x03) << 4)) - 32;

          float d1 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex);
          float d2 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 2L);
          float d3 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 4L);
          float d4 = d * qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 6L);

          out[outBase + index] = d1 * q1;
          out[outBase + index + 32] = d2 * q2;
          out[outBase + index + 64] = d3 * q3;
          out[outBase + index + 96] = d4 * q4;
        }
      }
    }
  }

  /** Batched row-major GEMV over GGUF Q4_0 rows. */
  default void ggufQ4_0MatVecDot(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    long rowBytes = ggufQ4_0RowBytes(cols);
    for (int row = 0; row < rows; row++) {
      out[row] = ggufQ4_0DotProduct(query, qWeight, row * rowBytes, cols);
    }
  }

  /** GGML-compatible Q4_0 by Q8_0 GEMV. The activation is quantized once and reused across rows. */
  default void ggufQ4_0Q8_0MatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ4_0RowBytes(cols);
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int row = 0; row < rows; row++) {
      out[row] = ggufQ4_0Q8_0ScalarRowDot(qWeight, row * rowBytes, blocks, q8Quants, q8Scales);
    }
  }

  /** Two Q4_0 projections sharing one Q8_0 activation quantization. */
  default void ggufQ4_0Q8_0DualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ4_0RowBytes(cols);
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int row = 0; row < firstRows; row++) {
      firstOut[row] =
          ggufQ4_0Q8_0ScalarRowDot(firstWeight, row * rowBytes, blocks, q8Quants, q8Scales);
    }
    for (int row = 0; row < secondRows; row++) {
      secondOut[row] =
          ggufQ4_0Q8_0ScalarRowDot(secondWeight, row * rowBytes, blocks, q8Quants, q8Scales);
    }
  }

  /** Three Q4_0 projections sharing one Q8_0 activation quantization. */
  default void ggufQ4_0Q8_0TripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ4_0RowBytes(cols);
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int row = 0; row < firstRows; row++) {
      firstOut[row] =
          ggufQ4_0Q8_0ScalarRowDot(firstWeight, row * rowBytes, blocks, q8Quants, q8Scales);
    }
    for (int row = 0; row < secondRows; row++) {
      secondOut[row] =
          ggufQ4_0Q8_0ScalarRowDot(secondWeight, row * rowBytes, blocks, q8Quants, q8Scales);
    }
    for (int row = 0; row < thirdRows; row++) {
      thirdOut[row] =
          ggufQ4_0Q8_0ScalarRowDot(thirdWeight, row * rowBytes, blocks, q8Quants, q8Scales);
    }
  }

  private static float ggufQ4_0Q8_0ScalarRowDot(
      MemorySegment qWeight, long rowOffset, int blocks, byte[] q8Quants, float[] q8Scales) {
    return ggufQ4_0Q8_0ScalarRowDot(qWeight, rowOffset, blocks, q8Quants, 0, q8Scales, 0);
  }

  private static float ggufQ4_0Q8_0ScalarRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      int quantBatchOffset,
      float[] q8Scales,
      int scaleBatchOffset) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
      float scale =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset))
              * q8Scales[scaleBatchOffset + block];
      long nibbleOffset = blockOffset + Short.BYTES;
      int quantOffset = quantBatchOffset + block * GGUF_Q_BLOCK_SIZE;
      int integerSum = 0;
      for (int index = 0; index < 16; index++) {
        int packed = qWeight.get(ValueLayout.JAVA_BYTE, nibbleOffset + index) & 0xFF;
        int lo = (packed & 0x0F) - 8;
        int hi = ((packed >>> 4) & 0x0F) - 8;
        integerSum += lo * q8Quants[quantOffset + index];
        integerSum += hi * q8Quants[quantOffset + index + 16];
      }
      sum = MathUtil.fma(scale, integerSum, sum);
    }
    return sum;
  }

  /** Q4_0 matrix multiplication over batch-major Q8_0-quantized activation rows. */
  default void ggufQ4_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      float[] laneScratch) {
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = ggufQ4_0RowBytes(cols);
    for (int batch = 0; batch < batchSize; batch++) {
      int quantBatchOffset = batch * cols;
      int scaleBatchOffset = batch * blocks;
      for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        long rowOffset = row * rowBytes;
        for (int block = 0; block < blocks; block++) {
          long blockOffset = rowOffset + (long) block * GGUF_Q4_0_BLOCK_BYTES;
          float scale =
              Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset))
                  * q8Scales[scaleBatchOffset + block];
          long nibbleOffset = blockOffset + Short.BYTES;
          int quantOffset = quantBatchOffset + block * GGUF_Q_BLOCK_SIZE;
          int integerSum = 0;
          for (int index = 0; index < 16; index++) {
            int packed = qWeight.get(ValueLayout.JAVA_BYTE, nibbleOffset + index) & 0xFF;
            int lo = (packed & 0x0F) - 8;
            int hi = ((packed >>> 4) & 0x0F) - 8;
            integerSum += lo * q8Quants[quantOffset + index];
            integerSum += hi * q8Quants[quantOffset + index + 16];
          }
          sum = MathUtil.fma(scale, integerSum, sum);
        }
        out[batch * rows + row] = sum;
      }
    }
  }

  /** Two Q4_0 projections over an activation batch sharing Q8_0 quantization and row dispatch. */
  default void ggufQ4_0Q8_0DualBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      float[] laneScratch) {
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = ggufQ4_0RowBytes(cols);
    int totalRows = Math.addExact(firstRows, secondRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        totalRows,
        cols,
        row -> {
          boolean first = row < firstRows;
          MemorySegment weight = first ? firstWeight : secondWeight;
          int matrixRow = first ? row : row - firstRows;
          int matrixRows = first ? firstRows : secondRows;
          float[] out = first ? firstOut : secondOut;
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * matrixRows + matrixRow] =
                ggufQ4_0Q8_0ScalarRowDot(
                    weight,
                    matrixRow * rowBytes,
                    blocks,
                    q8Quants,
                    batch * cols,
                    q8Scales,
                    batch * blocks);
          }
        });
  }

  /** Three Q4_0 projections over an activation batch sharing Q8_0 quantization and row dispatch. */
  default void ggufQ4_0Q8_0TripleBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      float[] laneScratch) {
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = ggufQ4_0RowBytes(cols);
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          MemorySegment weight;
          float[] out;
          int matrixRow;
          int matrixRows;
          if (row < secondStart) {
            weight = firstWeight;
            out = firstOut;
            matrixRow = row;
            matrixRows = firstRows;
          } else if (row < thirdStart) {
            weight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
            matrixRows = secondRows;
          } else {
            weight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
            matrixRows = thirdRows;
          }
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * matrixRows + matrixRow] =
                ggufQ4_0Q8_0ScalarRowDot(
                    weight,
                    matrixRow * rowBytes,
                    blocks,
                    q8Quants,
                    batch * cols,
                    q8Scales,
                    batch * blocks);
          }
        });
  }

  /** Batched row-major GEMV over GGUF Q4_K rows. */
  default void ggufQ4_KMatVecDot(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    long rowBytes = ggufQ4_KRowBytes(cols);
    for (int row = 0; row < rows; row++) {
      out[row] = ggufQ4_KDotProduct(query, qWeight, row * rowBytes, cols);
    }
  }

  /** GGML-compatible Q4_K by Q8_K GEMV. The activation is quantized once and reused across rows. */
  default void ggufQ4_KQ8_KMatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = ggufQ4_KRowBytes(cols);
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row ->
            out[row] =
                ggufQ4_KQ8_KScalarRowDot(
                    qWeight, row * rowBytes, blocks, q8Quants, q8Scales, q8Sums));
  }

  /** Q4_K matrix multiplication over batch-major Q8_K-quantized activation rows. */
  default void ggufQ4_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8Sums,
          batch * sumsPerBatch);
    }

    long rowBytes = ggufQ4_KRowBytes(cols);
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] =
                ggufQ4_KQ8_KScalarRowDot(
                    qWeight,
                    row * rowBytes,
                    blocks,
                    q8Quants,
                    batch * cols,
                    q8Scales,
                    batch * blocks,
                    q8Sums,
                    batch * sumsPerBatch);
          }
        });
  }

  /** Two Q4_K projections over an activation batch sharing Q8_K quantization and row dispatch. */
  default void ggufQ4_KQ8_KDualBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8Sums,
          batch * sumsPerBatch);
    }

    long rowBytes = ggufQ4_KRowBytes(cols);
    int totalRows = Math.addExact(firstRows, secondRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        totalRows,
        cols,
        row -> {
          boolean first = row < firstRows;
          MemorySegment weight = first ? firstWeight : secondWeight;
          int matrixRow = first ? row : row - firstRows;
          int matrixRows = first ? firstRows : secondRows;
          float[] out = first ? firstOut : secondOut;
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * matrixRows + matrixRow] =
                ggufQ4_KQ8_KScalarRowDot(
                    weight,
                    matrixRow * rowBytes,
                    blocks,
                    q8Quants,
                    batch * cols,
                    q8Scales,
                    batch * blocks,
                    q8Sums,
                    batch * sumsPerBatch);
          }
        });
  }

  /** Two Q4_K projections sharing one Q8_K activation quantization and row dispatch. */
  default void ggufQ4_KQ8_KDualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = ggufQ4_KRowBytes(cols);
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int totalRows = Math.addExact(firstRows, secondRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        totalRows,
        cols,
        row -> {
          boolean first = row < firstRows;
          MemorySegment weight = first ? firstWeight : secondWeight;
          int matrixRow = first ? row : row - firstRows;
          float[] out = first ? firstOut : secondOut;
          out[matrixRow] =
              ggufQ4_KQ8_KScalarRowDot(
                  weight, matrixRow * rowBytes, blocks, q8Quants, q8Scales, q8Sums);
        });
  }

  /** Three Q4_K projections sharing one Q8_K activation quantization and row dispatch. */
  default void ggufQ4_KQ8_KTripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = ggufQ4_KRowBytes(cols);
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          MemorySegment weight;
          float[] out;
          int matrixRow;
          if (row < secondStart) {
            weight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else if (row < thirdStart) {
            weight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
          } else {
            weight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
          }
          out[matrixRow] =
              ggufQ4_KQ8_KScalarRowDot(
                  weight, matrixRow * rowBytes, blocks, q8Quants, q8Scales, q8Sums);
        });
  }

  /** Two Q4_K projections and one Q6_K projection sharing Q8_K quantization and row dispatch. */
  default void ggufQ4_KQ4_KQ6_KQ8_KTripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long q4RowBytes = ggufQ4_KRowBytes(cols);
    long q6RowBytes = ggufQ6_KRowBytes(cols);
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          if (row < secondStart) {
            firstOut[row] =
                ggufQ4_KQ8_KScalarRowDot(
                    firstWeight, row * q4RowBytes, blocks, q8Quants, q8Scales, q8Sums);
          } else if (row < thirdStart) {
            int matrixRow = row - secondStart;
            secondOut[matrixRow] =
                ggufQ4_KQ8_KScalarRowDot(
                    secondWeight, matrixRow * q4RowBytes, blocks, q8Quants, q8Scales, q8Sums);
          } else {
            int matrixRow = row - thirdStart;
            thirdOut[matrixRow] =
                ggufQ6_KQ8_KScalarRowDot(
                    thirdWeight, matrixRow * q6RowBytes, blocks, q8Quants, 0, q8Scales, 0);
          }
        });
  }

  /**
   * Two Q4_K projections and one Q6_K projection over an activation batch sharing Q8_K quantization
   * and row dispatch.
   */
  default void ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
      float[] queries,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int batchSize,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    int blocks = cols / GGUF_Q4_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8Sums,
          batch * sumsPerBatch);
    }

    long q4RowBytes = ggufQ4_KRowBytes(cols);
    long q6RowBytes = ggufQ6_KRowBytes(cols);
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          if (row < secondStart) {
            for (int batch = 0; batch < batchSize; batch++) {
              firstOut[batch * firstRows + row] =
                  ggufQ4_KQ8_KScalarRowDot(
                      firstWeight,
                      row * q4RowBytes,
                      blocks,
                      q8Quants,
                      batch * cols,
                      q8Scales,
                      batch * blocks,
                      q8Sums,
                      batch * sumsPerBatch);
            }
          } else if (row < thirdStart) {
            int matrixRow = row - secondStart;
            for (int batch = 0; batch < batchSize; batch++) {
              secondOut[batch * secondRows + matrixRow] =
                  ggufQ4_KQ8_KScalarRowDot(
                      secondWeight,
                      matrixRow * q4RowBytes,
                      blocks,
                      q8Quants,
                      batch * cols,
                      q8Scales,
                      batch * blocks,
                      q8Sums,
                      batch * sumsPerBatch);
            }
          } else {
            int matrixRow = row - thirdStart;
            for (int batch = 0; batch < batchSize; batch++) {
              thirdOut[batch * thirdRows + matrixRow] =
                  ggufQ6_KQ8_KScalarRowDot(
                      thirdWeight,
                      matrixRow * q6RowBytes,
                      blocks,
                      q8Quants,
                      batch * cols,
                      q8Scales,
                      batch * blocks);
            }
          }
        });
  }

  private static float ggufQ4_KQ8_KScalarRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    return ggufQ4_KQ8_KScalarRowDot(
        qWeight, rowOffset, blocks, q8Quants, 0, q8Scales, 0, q8Sums, 0);
  }

  private static float ggufQ4_KQ8_KScalarRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      int quantBatchOffset,
      float[] q8Scales,
      int scaleBatchOffset,
      short[] q8Sums,
      int sumBatchOffset) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q4_K_BLOCK_BYTES;
      float q8Scale = q8Scales[scaleBatchOffset + block];
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
      float dMin =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES)) * q8Scale;
      long scalesOffset = blockOffset + GGUF_Q4_K_SCALES_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q4_K_QUANTS_OFFSET;
      int activationOffset = quantBatchOffset + block * GGUF_Q4_K_BLOCK_SIZE;
      int quantizedSum = 0;
      int minimumSum = 0;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int groupActivationOffset = activationOffset + group * 32;
        int groupDot = 0;
        for (int index = 0; index < 32; index++) {
          int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
          int quant = (packed >>> shift) & 0x0F;
          groupDot += quant * q8Quants[groupActivationOffset + index];
        }
        quantizedSum += scale * groupDot;
        int sumOffset =
            sumBatchOffset + (groupActivationOffset - quantBatchOffset) / GGUF_Q8_K_SUM_BLOCK_SIZE;
        minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
      }

      sum = MathUtil.fma(d, quantizedSum, sum);
      sum = MathUtil.fma(-dMin, minimumSum, sum);
    }
    return sum;
  }

  /** Batched row-major GEMV over GGUF Q5_K rows. */
  default void ggufQ5_KMatVecDot(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    long rowBytes = ggufQ5_KRowBytes(cols);
    for (int row = 0; row < rows; row++) {
      out[row] = ggufQ5_KDotProduct(query, qWeight, row * rowBytes, cols);
    }
  }

  /** GGML-compatible Q5_K by Q8_K GEMV. The activation is quantized once and reused across rows. */
  default void ggufQ5_KQ8_KMatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = ggufQ5_KRowBytes(cols);
    int blocks = cols / GGUF_Q5_K_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row ->
            out[row] =
                ggufQ5_KQ8_KScalarRowDot(
                    qWeight, row * rowBytes, blocks, q8Quants, 0, q8Scales, 0, q8Sums, 0));
  }

  /** Q5_K matrix multiplication over batch-major Q8_K-quantized activation rows. */
  default void ggufQ5_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    int blocks = cols / GGUF_Q5_K_BLOCK_SIZE;
    int sumsPerBatch = cols / GGUF_Q8_K_SUM_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries,
          batch * cols,
          cols,
          q8Quants,
          batch * cols,
          q8Scales,
          batch * blocks,
          q8Sums,
          batch * sumsPerBatch);
    }

    long rowBytes = ggufQ5_KRowBytes(cols);
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] =
                ggufQ5_KQ8_KScalarRowDot(
                    qWeight,
                    row * rowBytes,
                    blocks,
                    q8Quants,
                    batch * cols,
                    q8Scales,
                    batch * blocks,
                    q8Sums,
                    batch * sumsPerBatch);
          }
        });
  }

  /** Two Q5_K projections sharing one Q8_K activation quantization and row dispatch. */
  default void ggufQ5_KQ8_KDualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = ggufQ5_KRowBytes(cols);
    int blocks = cols / GGUF_Q5_K_BLOCK_SIZE;
    int totalRows = Math.addExact(firstRows, secondRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        totalRows,
        cols,
        row -> {
          boolean first = row < firstRows;
          MemorySegment weight = first ? firstWeight : secondWeight;
          int matrixRow = first ? row : row - firstRows;
          float[] out = first ? firstOut : secondOut;
          out[matrixRow] =
              ggufQ5_KQ8_KScalarRowDot(
                  weight, matrixRow * rowBytes, blocks, q8Quants, 0, q8Scales, 0, q8Sums, 0);
        });
  }

  /** Three Q5_K projections sharing one Q8_K activation quantization and row dispatch. */
  default void ggufQ5_KQ8_KTripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales, q8Sums);

    long rowBytes = ggufQ5_KRowBytes(cols);
    int blocks = cols / GGUF_Q5_K_BLOCK_SIZE;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        row -> {
          MemorySegment weight;
          float[] out;
          int matrixRow;
          if (row < secondStart) {
            weight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else if (row < thirdStart) {
            weight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
          } else {
            weight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
          }
          out[matrixRow] =
              ggufQ5_KQ8_KScalarRowDot(
                  weight, matrixRow * rowBytes, blocks, q8Quants, 0, q8Scales, 0, q8Sums, 0);
        });
  }

  private static float ggufQ5_KQ8_KScalarRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      int quantBatchOffset,
      float[] q8Scales,
      int scaleBatchOffset,
      short[] q8Sums,
      int sumBatchOffset) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q5_K_BLOCK_BYTES;
      float q8Scale = q8Scales[scaleBatchOffset + block];
      float d = Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scale;
      float dMin =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset + Short.BYTES)) * q8Scale;
      long scalesOffset = blockOffset + GGUF_Q5_K_SCALES_OFFSET;
      long highBitsOffset = blockOffset + GGUF_Q5_K_HIGH_BITS_OFFSET;
      long quantsOffset = blockOffset + GGUF_Q5_K_QUANTS_OFFSET;
      int activationOffset = block * GGUF_Q5_K_BLOCK_SIZE;
      int quantizedSum = 0;
      int minimumSum = 0;

      for (int group = 0; group < 8; group++) {
        int scale = GgufQuantizationSupport.qKScale(qWeight, scalesOffset, group);
        int min = GgufQuantizationSupport.qKMin(qWeight, scalesOffset, group);
        long packedOffset = quantsOffset + (long) (group >>> 1) * 32;
        int shift = (group & 1) * 4;
        int highBit = 1 << group;
        int groupActivationOffset = activationOffset + group * 32;
        int groupDot = 0;

        for (int index = 0; index < 32; index++) {
          int packed = qWeight.get(ValueLayout.JAVA_BYTE, packedOffset + index) & 0xFF;
          int highBits = qWeight.get(ValueLayout.JAVA_BYTE, highBitsOffset + index) & 0xFF;
          int quant = ((packed >>> shift) & 0x0F) | ((highBits & highBit) == 0 ? 0 : 16);
          groupDot += quant * q8Quants[quantBatchOffset + groupActivationOffset + index];
        }
        quantizedSum += scale * groupDot;
        int sumOffset = sumBatchOffset + groupActivationOffset / GGUF_Q8_K_SUM_BLOCK_SIZE;
        minimumSum += min * (q8Sums[sumOffset] + q8Sums[sumOffset + 1]);
      }

      sum = MathUtil.fma(d, quantizedSum, sum);
      sum = MathUtil.fma(-dMin, minimumSum, sum);
    }
    return sum;
  }

  /** Batched row-major GEMV over GGUF Q5_0 rows. */
  default void ggufQ5_0MatVecDot(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    long rowBytes = ggufQ5_0RowBytes(cols);
    for (int row = 0; row < rows; row++) {
      out[row] = ggufQ5_0DotProduct(query, qWeight, row * rowBytes, cols);
    }
  }

  /** GGML-compatible Q5_0 by Q8_0 GEMV. The activation is quantized once and reused across rows. */
  default void ggufQ5_0Q8_0MatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ5_0RowBytes(cols);
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          float sum = 0.0f;
          long rowOffset = row * rowBytes;
          for (int block = 0; block < blocks; block++) {
            long blockOffset = rowOffset + (long) block * GGUF_Q5_0_BLOCK_BYTES;
            float scale =
                Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset)) * q8Scales[block];
            int highBits = qWeight.get(GGUF_LE_INT, blockOffset + Short.BYTES);
            long quantsOffset = blockOffset + Short.BYTES + Integer.BYTES;
            int activationOffset = block * GGUF_Q_BLOCK_SIZE;
            int integerSum = 0;
            for (int index = 0; index < 16; index++) {
              int packed = qWeight.get(ValueLayout.JAVA_BYTE, quantsOffset + index) & 0xFF;
              int low = (packed & 0x0F) | (((highBits >>> index) & 1) << 4);
              int high = (packed >>> 4) | (((highBits >>> (index + 16)) & 1) << 4);
              integerSum += (low - 16) * q8Quants[activationOffset + index];
              integerSum += (high - 16) * q8Quants[activationOffset + index + 16];
            }
            sum = MathUtil.fma(scale, integerSum, sum);
          }
          out[row] = sum;
        });
  }

  /** Batched row-major GEMV over GGUF Q6_K rows. */
  default void ggufQ6_KMatVecDot(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    long rowBytes = ggufQ6_KRowBytes(cols);
    for (int row = 0; row < rows; row++) {
      out[row] = ggufQ6_KDotProduct(query, qWeight, row * rowBytes, cols);
    }
  }

  /** GGML-compatible Q6_K by Q8_K GEMV. The activation is quantized once and reused across rows. */
  default void ggufQ6_KQ8_KMatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_K(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ6_KRowBytes(cols);
    int blocks = cols / GGUF_Q6_K_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row ->
            out[row] =
                ggufQ6_KQ8_KScalarRowDot(
                    qWeight, row * rowBytes, blocks, q8Quants, 0, q8Scales, 0));
  }

  /** Q6_K matrix multiplication over batch-major Q8_K-quantized activation rows. */
  default void ggufQ6_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    int blocks = cols / GGUF_Q6_K_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_K(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks, null, 0);
    }

    long rowBytes = ggufQ6_KRowBytes(cols);
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        row -> {
          for (int batch = 0; batch < batchSize; batch++) {
            out[batch * rows + row] =
                ggufQ6_KQ8_KScalarRowDot(
                    qWeight,
                    row * rowBytes,
                    blocks,
                    q8Quants,
                    batch * cols,
                    q8Scales,
                    batch * blocks);
          }
        });
  }

  private static float ggufQ6_KQ8_KScalarRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      int quantBatchOffset,
      float[] q8Scales,
      int scaleBatchOffset) {
    GgufQuantizationSupport.Q6Scratch scratch = GgufQuantizationSupport.q6Scratch();
    int[] integerSums = scratch.integerSums;
    float[] laneSums = scratch.laneSums;
    java.util.Arrays.fill(laneSums, 0.0f);

    for (int block = 0; block < blocks; block++) {
      java.util.Arrays.fill(integerSums, 0);
      long blockOffset = rowOffset + (long) block * GGUF_Q6_K_BLOCK_BYTES;
      float d =
          Float.float16ToFloat(
                  qWeight.get(
                      GGUF_LE_SHORT,
                      blockOffset + GGUF_Q6_K_QL_BYTES + GGUF_Q6_K_QH_BYTES + GGUF_Q6_K_SCALES))
              * q8Scales[scaleBatchOffset + block];
      long qlOffset = blockOffset;
      long qhOffset = blockOffset + GGUF_Q6_K_QL_BYTES;
      long scaleOffset = qhOffset + GGUF_Q6_K_QH_BYTES;
      int queryOffset = quantBatchOffset + block * GGUF_Q6_K_BLOCK_SIZE;

      for (int superBlock = 0; superBlock < 2; superBlock++) {
        long qlBase = qlOffset + (long) superBlock * 64;
        long qhBase = qhOffset + (long) superBlock * 32;
        long scaleBase = scaleOffset + (long) superBlock * 8;
        int quantBase = queryOffset + superBlock * 128;

        for (int index = 0; index < 32; index++) {
          int scaleIndex = index / 16;
          int ql1 = qWeight.get(ValueLayout.JAVA_BYTE, qlBase + index) & 0xFF;
          int ql2 = qWeight.get(ValueLayout.JAVA_BYTE, qlBase + 32L + index) & 0xFF;
          int qh = qWeight.get(ValueLayout.JAVA_BYTE, qhBase + index) & 0xFF;
          int q1 = ((ql1 & 0x0F) | ((qh & 0x03) << 4)) - 32;
          int q2 = ((ql2 & 0x0F) | (((qh >>> 2) & 0x03) << 4)) - 32;
          int q3 = ((ql1 >>> 4) | (((qh >>> 4) & 0x03) << 4)) - 32;
          int q4 = ((ql2 >>> 4) | (((qh >>> 6) & 0x03) << 4)) - 32;
          int s1 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex);
          int s2 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 2L);
          int s3 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 4L);
          int s4 = qWeight.get(ValueLayout.JAVA_BYTE, scaleBase + scaleIndex + 6L);
          int lane = index & 7;

          integerSums[lane] += s1 * q1 * q8Quants[quantBase + index];
          integerSums[lane] += s2 * q2 * q8Quants[quantBase + index + 32];
          integerSums[lane] += s3 * q3 * q8Quants[quantBase + index + 64];
          integerSums[lane] += s4 * q4 * q8Quants[quantBase + index + 96];
        }
      }

      for (int lane = 0; lane < laneSums.length; lane++) {
        laneSums[lane] = MathUtil.fma(d, integerSums[lane], laneSums[lane]);
      }
    }

    float sum = 0.0f;
    for (float laneSum : laneSums) {
      sum += laneSum;
    }
    return sum;
  }

  /** Batched row-major GEMV over GGUF Q8_0 rows. */
  default void ggufQ8_0MatVecDot(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    long rowBytes = ggufQ8_0RowBytes(cols);
    for (int row = 0; row < rows; row++) {
      out[row] = ggufQ8_0DotProduct(query, qWeight, row * rowBytes, cols);
    }
  }

  /** GGML-compatible Q8_0 by Q8_0 GEMV. The activation is quantized once and reused across rows. */
  default void ggufQ8_0Q8_0MatVecDot(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ8_0RowBytes(cols);
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    GgufParallelSupport.forEachRow(
        qWeight,
        rows,
        cols,
        GgufParallelSupport.Q8_MIN_ELEMENTS,
        row ->
            out[row] =
                ggufQ8_0Q8_0ScalarRowDot(qWeight, row * rowBytes, blocks, q8Quants, q8Scales));
  }

  /** Q8_0 matrix multiplication over batch-major Q8_0-quantized activation rows. */
  default void ggufQ8_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    for (int batch = 0; batch < batchSize; batch++) {
      GgufQuantizationSupport.quantizeQ8_0(
          queries, batch * cols, cols, q8Quants, batch * cols, q8Scales, batch * blocks);
    }

    long rowBytes = ggufQ8_0RowBytes(cols);
    for (int batch = 0; batch < batchSize; batch++) {
      int quantBatchOffset = batch * cols;
      int scaleBatchOffset = batch * blocks;
      for (int row = 0; row < rows; row++) {
        out[batch * rows + row] =
            ggufQ8_0Q8_0ScalarRowDot(
                qWeight,
                row * rowBytes,
                blocks,
                q8Quants,
                quantBatchOffset,
                q8Scales,
                scaleBatchOffset);
      }
    }
  }

  /** Two Q8_0 projections sharing one Q8_0 activation quantization and row dispatch. */
  default void ggufQ8_0Q8_0DualMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ8_0RowBytes(cols);
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    int totalRows = Math.addExact(firstRows, secondRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        totalRows,
        cols,
        GgufParallelSupport.Q8_MIN_ELEMENTS,
        row -> {
          boolean first = row < firstRows;
          MemorySegment weight = first ? firstWeight : secondWeight;
          int matrixRow = first ? row : row - firstRows;
          float[] out = first ? firstOut : secondOut;
          out[matrixRow] =
              ggufQ8_0Q8_0ScalarRowDot(weight, matrixRow * rowBytes, blocks, q8Quants, q8Scales);
        });
  }

  /** Three Q8_0 projections sharing one Q8_0 activation quantization and row dispatch. */
  default void ggufQ8_0Q8_0TripleMatVecDot(
      float[] query,
      MemorySegment firstWeight,
      int firstRows,
      float[] firstOut,
      MemorySegment secondWeight,
      int secondRows,
      float[] secondOut,
      MemorySegment thirdWeight,
      int thirdRows,
      float[] thirdOut,
      int cols,
      byte[] q8Quants,
      float[] q8Scales) {
    quantizeQ8_0(query, cols, q8Quants, q8Scales);

    long rowBytes = ggufQ8_0RowBytes(cols);
    int blocks = cols / GGUF_Q_BLOCK_SIZE;
    int secondStart = firstRows;
    int thirdStart = Math.addExact(firstRows, secondRows);
    int totalRows = Math.addExact(thirdStart, thirdRows);
    GgufParallelSupport.forEachRow(
        firstWeight,
        secondWeight,
        thirdWeight,
        totalRows,
        cols,
        GgufParallelSupport.Q8_MIN_ELEMENTS,
        row -> {
          MemorySegment weight;
          float[] out;
          int matrixRow;
          if (row < secondStart) {
            weight = firstWeight;
            out = firstOut;
            matrixRow = row;
          } else if (row < thirdStart) {
            weight = secondWeight;
            out = secondOut;
            matrixRow = row - secondStart;
          } else {
            weight = thirdWeight;
            out = thirdOut;
            matrixRow = row - thirdStart;
          }
          out[matrixRow] =
              ggufQ8_0Q8_0ScalarRowDot(weight, matrixRow * rowBytes, blocks, q8Quants, q8Scales);
        });
  }

  private static float ggufQ8_0Q8_0ScalarRowDot(
      MemorySegment qWeight, long rowOffset, int blocks, byte[] q8Quants, float[] q8Scales) {
    return ggufQ8_0Q8_0ScalarRowDot(qWeight, rowOffset, blocks, q8Quants, 0, q8Scales, 0);
  }

  private static float ggufQ8_0Q8_0ScalarRowDot(
      MemorySegment qWeight,
      long rowOffset,
      int blocks,
      byte[] q8Quants,
      int quantBaseOffset,
      float[] q8Scales,
      int scaleBaseOffset) {
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      long blockOffset = rowOffset + (long) block * GGUF_Q8_0_BLOCK_BYTES;
      float scale =
          Float.float16ToFloat(qWeight.get(GGUF_LE_SHORT, blockOffset))
              * q8Scales[scaleBaseOffset + block];
      long quantOffset = blockOffset + Short.BYTES;
      int queryOffset = quantBaseOffset + block * GGUF_Q_BLOCK_SIZE;
      int integerSum = 0;
      for (int index = 0; index < GGUF_Q_BLOCK_SIZE; index++) {
        integerSum +=
            qWeight.get(ValueLayout.JAVA_BYTE, quantOffset + index) * q8Quants[queryOffset + index];
      }
      sum = MathUtil.fma(scale, integerSum, sum);
    }
    return sum;
  }

  /**
   * Fused matrix-vector squared L2 distance: fills {@code out[i] = squaredL2(query, matrix[i])} for
   * {@code i in [0, numRows)}.
   *
   * <p>The default implementation is a simple loop. SIMD subclasses override this with 4-row
   * unrolled accumulation.
   *
   * @param query the query vector
   * @param matrix the matrix rows
   * @param out the output array (must have length &ge; {@code numRows})
   * @param numRows the number of rows to process
   */
  default void matVecSquaredL2(float[] query, float[][] matrix, float[] out, int numRows) {
    for (int i = 0; i < numRows; i++) {
      out[i] = squareDistance(query, 0, matrix[i], 0, query.length);
    }
  }

  /**
   * Off-heap fused matrix-vector dot product: fills {@code out[i] = dot(query, rows[i])} for {@code
   * i in [0, count)}, where each {@code rows[i]} is a {@link MemorySegment} holding {@code dim}
   * little-endian float32s (typically a zero-copy mmap/off-heap slice).
   *
   * <p>This is the segment-scoring analogue of {@link #matVecDot(float[], float[][], float[],
   * int)}: the query is still an on-heap {@code float[]} (uploaded once by the caller); only the
   * matrix rows come from segments. The default implementation is a scalar per-row loop. SIMD
   * subclasses override this to load each query SIMD chunk <em>once</em> and apply it to 4 segment
   * rows simultaneously, giving the zero-copy path the same 4× query-load amortization as the
   * {@code float[][]} path.
   *
   * @param query the query vector (length = {@code dim})
   * @param rows the matrix rows as off-heap segments (each holds {@code dim} little-endian floats)
   * @param dim the number of float elements per row
   * @param out the output array (must have length &ge; {@code count})
   * @param count the number of rows to process (must be &le; {@code rows.length})
   */
  default void matVecDot(float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    for (int i = 0; i < count; i++) {
      MemorySegment row = rows[i];
      float s = 0f;
      for (int d = 0; d < dim; d++) {
        s = MathUtil.fma(query[d], row.getAtIndex(ValueLayout.JAVA_FLOAT, d), s);
      }
      out[i] = s;
    }
  }

  /**
   * Off-heap fused matrix-vector squared L2 distance: fills {@code out[i] = squaredL2(query,
   * rows[i])} for {@code i in [0, count)}. Segment-scoring analogue of {@link
   * #matVecSquaredL2(float[], float[][], float[], int)}; see {@link #matVecDot(float[],
   * MemorySegment[], int, float[], int)} for the query-load amortization rationale.
   *
   * @param query the query vector
   * @param rows the matrix rows as off-heap segments
   * @param dim the number of float elements per row
   * @param out the output array (must have length &ge; {@code count})
   * @param count the number of rows to process
   */
  default void matVecSquaredL2(
      float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    for (int i = 0; i < count; i++) {
      MemorySegment row = rows[i];
      float s = 0f;
      for (int d = 0; d < dim; d++) {
        float e = query[d] - row.getAtIndex(ValueLayout.JAVA_FLOAT, d);
        s = MathUtil.fma(e, e, s);
      }
      out[i] = s;
    }
  }

  /**
   * Fused matrix-vector cosine similarity: fills {@code out[i] = cosine(query, matrix[i])} for
   * {@code i in [0, count)}, returning the RAW cosine value in {@code [-1, 1]} (callers apply the
   * {@code (1+cos)/2} score transform).
   *
   * <p>The query norm {@code ‖query‖²} is constant for the whole batch, so it is computed once
   * (single reduction over {@code query}); only the per-row dot product and row norm vary. The
   * default implementation is a per-row loop over {@link #cosine(float[], float[])}. SIMD
   * subclasses override this to load each query SIMD chunk <em>once</em> per 4-row group and
   * accumulate {@code dot[r]} and {@code ‖row[r]‖²} for 4 rows simultaneously — the
   * query-load-amortized analogue of {@link #matVecDot(float[], float[][], float[], int)}, carrying
   * a second (row-norm) accumulator set plus the single shared query norm.
   *
   * <p>Zero-vector behavior matches {@link #cosine(float[], float[])}: a zero row (or zero query)
   * yields {@code 0/0 = NaN}, identical to the per-row reference.
   *
   * @param query the query vector (length = {@code matrix[0].length})
   * @param rows the matrix rows (each row must have the same length as {@code query})
   * @param out the output array (must have length &ge; {@code count})
   * @param count the number of rows to process (must be &le; {@code rows.length})
   */
  default void batchCosine(float[] query, float[][] rows, float[] out, int count) {
    for (int i = 0; i < count; i++) {
      out[i] = cosine(query, rows[i]);
    }
  }

  /**
   * Off-heap fused matrix-vector cosine similarity: fills {@code out[i] = cosine(query, rows[i])}
   * for {@code i in [0, count)}, where each {@code rows[i]} is a {@link MemorySegment} holding
   * {@code dim} little-endian float32s (typically a zero-copy mmap/off-heap slice). Returns RAW
   * cosine values in {@code [-1, 1]}.
   *
   * <p>Segment-scoring analogue of {@link #batchCosine(float[], float[][], float[], int)}: the
   * query stays an on-heap {@code float[]} (its norm computed once); only the matrix rows come from
   * segments. The default implementation is a scalar per-row loop. SIMD subclasses override with
   * the 4-row fused kernel.
   *
   * @param query the query vector (length = {@code dim})
   * @param rows the matrix rows as off-heap segments (each holds {@code dim} little-endian floats)
   * @param dim the number of float elements per row
   * @param out the output array (must have length &ge; {@code count})
   * @param count the number of rows to process (must be &le; {@code rows.length})
   */
  default void batchCosine(float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    for (int i = 0; i < count; i++) {
      MemorySegment row = rows[i];
      float dot = 0f;
      float qn = 0f;
      float rn = 0f;
      for (int d = 0; d < dim; d++) {
        float q = query[d];
        float rv = row.getAtIndex(ValueLayout.JAVA_FLOAT, d);
        dot = MathUtil.fma(q, rv, dot);
        qn = MathUtil.fma(q, q, qn);
        rn = MathUtil.fma(rv, rv, rn);
      }
      out[i] = (float) (dot / Math.sqrt((double) qn * (double) rn));
    }
  }

  // --- PQ ADC (Asymmetric Distance Computation) kernels ---

  /**
   * Sums one entry per subspace from a precomputed query→centroid table, indexed by PQ codes.
   *
   * <p>Computes {@code sum_{m=0..M-1} table[m][codes[codesOffset + m] & 0xFF]} — the core inner
   * loop of ADC scoring against PQ-compressed vectors. Equivalent to the lookup {@code T[m][c_m]}
   * pattern used by JVector's {@code VectorUtil.assembleAndSum}.
   *
   * @param table precomputed table of shape {@code [numSubspaces][numClusters]}; typically produced
   *     by a product-quantizer ADC table builder
   * @param codes byte buffer holding at least {@code numSubspaces} PQ codes starting at {@code
   *     codesOffset} (unsigned)
   * @param codesOffset starting offset in {@code codes}
   * @param numSubspaces M — the number of PQ subspaces
   * @return the accumulated partial distance/similarity (unscored raw sum)
   */
  default float assembleAndSum(float[][] table, byte[] codes, int codesOffset, int numSubspaces) {
    float sum = 0f;
    for (int m = 0; m < numSubspaces; m++) {
      sum += table[m][codes[codesOffset + m] & 0xFF];
    }
    return sum;
  }

  private static void checkGgufBlockAligned(int dimensions) {
    if (dimensions % GGUF_Q_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "GGUF quantized dimensions must be a multiple of 32: " + dimensions);
    }
  }

  private static void checkGgufQ6_KBlockAligned(int dimensions) {
    if (dimensions % GGUF_Q6_K_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "GGUF Q6_K dimensions must be a multiple of 256: " + dimensions);
    }
  }

  private static void checkGgufQ4_KBlockAligned(int dimensions) {
    if (dimensions % GGUF_Q4_K_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "GGUF Q4_K dimensions must be a multiple of 256: " + dimensions);
    }
  }

  private static void checkGgufQ5_KBlockAligned(int dimensions) {
    if (dimensions % GGUF_Q5_K_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "GGUF Q5_K dimensions must be a multiple of 256: " + dimensions);
    }
  }

  private static long ggufQ4_0RowBytes(int dimensions) {
    checkGgufBlockAligned(dimensions);
    return (long) (dimensions / GGUF_Q_BLOCK_SIZE) * GGUF_Q4_0_BLOCK_BYTES;
  }

  private static long ggufQ5_0RowBytes(int dimensions) {
    checkGgufBlockAligned(dimensions);
    return (long) (dimensions / GGUF_Q_BLOCK_SIZE) * GGUF_Q5_0_BLOCK_BYTES;
  }

  private static long ggufQ8_0RowBytes(int dimensions) {
    checkGgufBlockAligned(dimensions);
    return (long) (dimensions / GGUF_Q_BLOCK_SIZE) * GGUF_Q8_0_BLOCK_BYTES;
  }

  private static long ggufQ4_KRowBytes(int dimensions) {
    checkGgufQ4_KBlockAligned(dimensions);
    return (long) (dimensions / GGUF_Q4_K_BLOCK_SIZE) * GGUF_Q4_K_BLOCK_BYTES;
  }

  private static long ggufQ5_KRowBytes(int dimensions) {
    checkGgufQ5_KBlockAligned(dimensions);
    return (long) (dimensions / GGUF_Q5_K_BLOCK_SIZE) * GGUF_Q5_K_BLOCK_BYTES;
  }

  private static long ggufQ6_KRowBytes(int dimensions) {
    checkGgufQ6_KBlockAligned(dimensions);
    return (long) (dimensions / GGUF_Q6_K_BLOCK_SIZE) * GGUF_Q6_K_BLOCK_BYTES;
  }

  private static void quantizeQ8_K(float[] query, int dimensions, byte[] quants, float[] scales) {
    GgufQuantizationSupport.quantizeQ8_K(query, dimensions, quants, scales, null);
  }

  private static void quantizeQ8_K(
      float[] query, int dimensions, byte[] quants, float[] scales, short[] sums) {
    GgufQuantizationSupport.quantizeQ8_K(query, dimensions, quants, scales, sums);
  }

  private static void quantizeQ8_0(float[] query, int dimensions, byte[] quants, float[] scales) {
    GgufQuantizationSupport.quantizeQ8_0(query, dimensions, quants, scales);
  }

  /**
   * Batched variant of {@link #assembleAndSum} over {@code count} neighbors stored in a packed byte
   * buffer. Each neighbor occupies {@code numSubspaces} consecutive bytes: neighbor {@code i}
   * starts at {@code codesOffset + i * numSubspaces}.
   *
   * <p>The 4-row unrolled default keeps each {@code table[m]} pointer hot across four lookups,
   * amortising the subspace-table load and eliminating three redundant table row references per
   * iteration vs the scalar scorer.
   */
  default void batchAssembleAndSum(
      float[][] table,
      byte[] packedCodes,
      int codesOffset,
      float[] out,
      int count,
      int numSubspaces) {
    int i = 0;
    int m;
    int rowGroup = count & ~3;
    for (; i < rowGroup; i += 4) {
      float s0 = 0f;
      float s1 = 0f;
      float s2 = 0f;
      float s3 = 0f;
      int o0 = codesOffset + i * numSubspaces;
      int o1 = o0 + numSubspaces;
      int o2 = o1 + numSubspaces;
      int o3 = o2 + numSubspaces;
      for (m = 0; m < numSubspaces; m++) {
        float[] tm = table[m];
        s0 += tm[packedCodes[o0 + m] & 0xFF];
        s1 += tm[packedCodes[o1 + m] & 0xFF];
        s2 += tm[packedCodes[o2 + m] & 0xFF];
        s3 += tm[packedCodes[o3 + m] & 0xFF];
      }
      out[i] = s0;
      out[i + 1] = s1;
      out[i + 2] = s2;
      out[i + 3] = s3;
    }
    for (; i < count; i++) {
      float s = 0f;
      int o = codesOffset + i * numSubspaces;
      for (m = 0; m < numSubspaces; m++) {
        s += table[m][packedCodes[o + m] & 0xFF];
      }
      out[i] = s;
    }
  }
}
