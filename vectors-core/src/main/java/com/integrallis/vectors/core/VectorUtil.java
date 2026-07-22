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

/**
 * Public API facade for vector distance and similarity operations. Delegates to the best available
 * {@link VectorUtilSupport} implementation (SIMD or scalar).
 *
 * <p>All methods validate preconditions (matching dimensions) and assert result sanity. This is the
 * primary entry point for application code.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * float[] a = {1.0f, 2.0f, 3.0f};
 * float[] b = {4.0f, 5.0f, 6.0f};
 * float dot = VectorUtil.dotProduct(a, b);
 * float l2 = VectorUtil.squareDistance(a, b);
 * float cos = VectorUtil.cosine(a, b);
 * }</pre>
 */
public final class VectorUtil {

  private static final VectorUtilSupport IMPL = VectorizationProvider.getInstance();

  private VectorUtil() {}

  /** Returns the SIMD and GGUF execution capabilities selected by vectors-core. */
  public static VectorRuntimeCapabilities runtimeCapabilities() {
    return VectorizationProvider.runtimeCapabilities();
  }

  // --- Float distance operations ---

  /** Computes the dot product of two float vectors of equal length. */
  public static float dotProduct(float[] a, float[] b) {
    checkDimensions(a.length, b.length);
    float r = IMPL.dotProduct(a, b);
    assert Float.isFinite(r);
    return r;
  }

  /** Computes the dot product of sub-vectors at the given offsets and length. */
  public static float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length) {
    return IMPL.dotProduct(a, aOffset, b, bOffset, length);
  }

  /** Computes the squared Euclidean (L2) distance between two float vectors. */
  public static float squareDistance(float[] a, float[] b) {
    checkDimensions(a.length, b.length);
    float r = IMPL.squareDistance(a, b);
    assert Float.isFinite(r) && r >= 0f;
    return r;
  }

  /** Computes the squared L2 distance of sub-vectors at the given offsets. */
  public static float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int length) {
    return IMPL.squareDistance(a, aOffset, b, bOffset, length);
  }

  /** Computes the cosine similarity between two float vectors. Returns value in [-1, 1]. */
  public static float cosine(float[] a, float[] b) {
    checkDimensions(a.length, b.length);
    return IMPL.cosine(a, b);
  }

  // --- Byte distance operations ---

  /** Computes the dot product of two signed byte vectors. */
  public static int dotProduct(byte[] a, byte[] b) {
    checkDimensions(a.length, b.length);
    return IMPL.dotProduct(a, b);
  }

  /** Computes the squared L2 distance between two signed byte vectors. */
  public static int squareDistance(byte[] a, byte[] b) {
    checkDimensions(a.length, b.length);
    return IMPL.squareDistance(a, b);
  }

  /** Computes the cosine similarity between two signed byte vectors. */
  public static float cosine(byte[] a, byte[] b) {
    checkDimensions(a.length, b.length);
    return IMPL.cosine(a, b);
  }

  // --- MemorySegment distance operations ---

  /** Computes the dot product from off-heap float vectors stored in MemorySegments. */
  public static float dotProduct(MemorySegment a, MemorySegment b, int dimensions) {
    return IMPL.dotProduct(a, b, dimensions);
  }

  /** Computes the squared L2 distance from off-heap float vectors stored in MemorySegments. */
  public static float squareDistance(MemorySegment a, MemorySegment b, int dimensions) {
    return IMPL.squareDistance(a, b, dimensions);
  }

  /** Computes the cosine similarity from off-heap float vectors stored in MemorySegments. */
  public static float cosine(MemorySegment a, MemorySegment b, int dimensions) {
    return IMPL.cosine(a, b, dimensions);
  }

  // --- Binary distance ---

  /** Computes the Hamming distance between two binary vectors stored as packed long arrays. */
  public static int hammingDistance(long[] a, long[] b) {
    checkDimensions(a.length, b.length);
    return IMPL.hammingDistance(a, b);
  }

  // --- Vector arithmetic ---

  /** Adds v2 to v1 element-wise in place. */
  public static void addInPlace(float[] v1, float[] v2) {
    checkDimensions(v1.length, v2.length);
    IMPL.addInPlace(v1, v2);
  }

  /** Adds {@code vector[vectorOffset..] * scale} into {@code out[outOffset..]} in place. */
  public static void addScaledInPlace(
      float[] out, int outOffset, float[] vector, int vectorOffset, int length, float scale) {
    checkSubVectorArguments(out, outOffset, vector, vectorOffset, length);
    IMPL.addScaledInPlace(out, outOffset, vector, vectorOffset, length, scale);
  }

  /**
   * Adds a weighted sum of flat, strided matrix rows to {@code out} in ascending row order. Inputs
   * must not alias {@code out}.
   */
  public static void addWeightedRowsInPlace(
      float[] out,
      int outOffset,
      float[] matrix,
      int matrixOffset,
      int rowStride,
      float[] weights,
      int weightsOffset,
      int rows,
      int columns) {
    checkWeightedRowsArguments(
        out, outOffset, matrix, matrixOffset, rowStride, weights, weightsOffset, rows, columns);
    IMPL.addWeightedRowsInPlace(
        out, outOffset, matrix, matrixOffset, rowStride, weights, weightsOffset, rows, columns);
  }

  /** Subtracts v2 from v1 element-wise in place. */
  public static void subInPlace(float[] v1, float[] v2) {
    checkDimensions(v1.length, v2.length);
    IMPL.subInPlace(v1, v2);
  }

  /** Scales each element of the vector by the given multiplier. */
  public static void scale(float[] vector, float multiplier) {
    IMPL.scale(vector, multiplier);
  }

  /** Returns the sum of all elements in the vector. */
  public static float sum(float[] vector) {
    return IMPL.sum(vector);
  }

  // --- Batch distance operations (used by CentroidIndex for centroid scoring) ---

  /**
   * Fills {@code out[i]} with the dot product of {@code query} and {@code matrix[i]} for all rows
   * {@code i} in {@code [0, matrix.length)}. {@code out.length} must be &ge; {@code matrix.length}.
   *
   * <p>Delegates to {@link VectorUtilSupport#matVecDot}, which SIMD implementations override with a
   * fused 4-row kernel that loads each query SIMD chunk once and applies it to 4 rows
   * simultaneously, reducing query memory traffic by 4×.
   */
  public static void batchDotProduct(float[] query, float[][] matrix, float[] out) {
    checkBatchArguments(query, matrix, out, matrix.length);
    IMPL.matVecDot(query, matrix, out, matrix.length);
  }

  /**
   * Fused GEMV dot product over the first {@code numRows} rows of {@code matrix}. Useful when
   * {@code matrix} is a reusable scratch buffer sized for a maximum batch but the current call
   * processes only a prefix.
   */
  public static void batchDotProduct(float[] query, float[][] matrix, float[] out, int numRows) {
    checkBatchArguments(query, matrix, out, numRows);
    IMPL.matVecDot(query, matrix, out, numRows);
  }

  /**
   * Fused GEMV dot product over a flat row-major matrix. Fills {@code out[row]} with the dot
   * product of {@code query} and {@code rowMajorMatrix[row * cols .. row * cols + cols)}.
   *
   * <p>This is the preferred call shape for dense inference tensors because it avoids materializing
   * {@code float[][]} row wrappers around contiguous row-major weights.
   */
  public static void batchDotProduct(
      float[] query, float[] rowMajorMatrix, int rows, int cols, float[] out) {
    checkRowMajorBatchArguments(query, rowMajorMatrix, rows, cols, out);
    IMPL.matVecDot(query, rowMajorMatrix, rows, cols, out);
  }

  /**
   * Fills an output range from offset query and strided matrix rows. Every result is bit-identical
   * to calling this provider's offset {@link #dotProduct(float[], int, float[], int, int)} for that
   * row independently.
   *
   * <p>The output must not alias either input array.
   */
  public static void batchDotProductExact(
      float[] query,
      int queryOffset,
      float[] matrix,
      int matrixOffset,
      int rowStride,
      int rows,
      int columns,
      float[] out,
      int outOffset) {
    checkExactStridedBatchArguments(
        query, queryOffset, matrix, matrixOffset, rowStride, rows, columns, out, outOffset);
    IMPL.matVecDotExact(
        query, queryOffset, matrix, matrixOffset, rowStride, rows, columns, out, outOffset);
  }

  /**
   * Fused GEMV over a little-endian GGUF F32 matrix stored in mapped or off-heap memory.
   *
   * <p>This avoids copying the full matrix to a temporary heap array before each projection.
   */
  public static void ggufF32BatchDotProduct(
      float[] query, MemorySegment weight, int rows, int cols, float[] out) {
    checkGgufF32BatchArguments(query, weight, rows, cols, out);
    IMPL.ggufF32MatVecDot(query, weight, rows, cols, out);
  }

  /**
   * Dot product of a full-precision query with one GGUF Q4_0 quantized row.
   *
   * <p>The operation fuses Q4_0 dequantization and dot-product accumulation without allocating a
   * temporary decoded row.
   */
  public static float ggufQ4_0DotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQuantizedDotArguments(
        query, qWeight, byteOffset, dimensions, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    return IMPL.ggufQ4_0DotProduct(query, qWeight, byteOffset, dimensions);
  }

  /**
   * Dot product of a full-precision query with one GGUF Q4_K quantized row.
   *
   * <p>The operation fuses Q4_K dequantization and dot-product accumulation without allocating a
   * temporary decoded row.
   */
  public static float ggufQ4_KDotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQuantizedDotArguments(
        query,
        qWeight,
        byteOffset,
        dimensions,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    return IMPL.ggufQ4_KDotProduct(query, qWeight, byteOffset, dimensions);
  }

  /** Dequantizes GGUF Q4_K blocks into a caller-owned float array. */
  public static void ggufQ4_KDequantize(
      MemorySegment qWeight, long byteOffset, float[] out, int outOffset, int dimensions) {
    Objects.requireNonNull(qWeight, "qWeight");
    Objects.requireNonNull(out, "out");
    if (byteOffset < 0) {
      throw new IllegalArgumentException("byteOffset must be >= 0: " + byteOffset);
    }
    if (outOffset < 0 || dimensions < 0 || outOffset > out.length - dimensions) {
      throw new IllegalArgumentException(
          "out range is invalid: offset="
              + outOffset
              + ", dimensions="
              + dimensions
              + ", length="
              + out.length);
    }
    if (dimensions % VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "GGUF Q4_K dimensions must be a multiple of 256: " + dimensions);
    }
    long required =
        byteOffset
            + ggufQuantizedRowBytes(
                dimensions,
                VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
                VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    if (required < byteOffset || required > qWeight.byteSize()) {
      throw new IllegalArgumentException(
          "qWeight byteSize is too small for requested values: "
              + qWeight.byteSize()
              + " < "
              + required);
    }
    IMPL.ggufQ4_KDequantize(qWeight, byteOffset, out, outOffset, dimensions);
  }

  /** Dot product of a full-precision query with one GGUF Q5_K quantized row. */
  public static float ggufQ5_KDotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQuantizedDotArguments(
        query,
        qWeight,
        byteOffset,
        dimensions,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    return IMPL.ggufQ5_KDotProduct(query, qWeight, byteOffset, dimensions);
  }

  /** Dequantizes GGUF Q5_K blocks into a caller-owned float array. */
  public static void ggufQ5_KDequantize(
      MemorySegment qWeight, long byteOffset, float[] out, int outOffset, int dimensions) {
    Objects.requireNonNull(qWeight, "qWeight");
    Objects.requireNonNull(out, "out");
    if (byteOffset < 0) {
      throw new IllegalArgumentException("byteOffset must be >= 0: " + byteOffset);
    }
    if (outOffset < 0 || dimensions < 0 || outOffset > out.length - dimensions) {
      throw new IllegalArgumentException(
          "out range is invalid: offset="
              + outOffset
              + ", dimensions="
              + dimensions
              + ", length="
              + out.length);
    }
    if (dimensions % VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "GGUF Q5_K dimensions must be a multiple of 256: " + dimensions);
    }
    long required =
        byteOffset
            + ggufQuantizedRowBytes(
                dimensions,
                VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
                VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    if (required < byteOffset || required > qWeight.byteSize()) {
      throw new IllegalArgumentException(
          "qWeight byteSize is too small for requested values: "
              + qWeight.byteSize()
              + " < "
              + required);
    }
    IMPL.ggufQ5_KDequantize(qWeight, byteOffset, out, outOffset, dimensions);
  }

  /** Dot product of a full-precision query with one GGUF Q5_0 quantized row. */
  public static float ggufQ5_0DotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQuantizedDotArguments(
        query, qWeight, byteOffset, dimensions, VectorUtilSupport.GGUF_Q5_0_BLOCK_BYTES);
    return IMPL.ggufQ5_0DotProduct(query, qWeight, byteOffset, dimensions);
  }

  /**
   * Dot product of a full-precision query with one GGUF Q8_0 quantized row.
   *
   * <p>The operation fuses Q8_0 dequantization and dot-product accumulation without allocating a
   * temporary decoded row.
   */
  public static float ggufQ8_0DotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQuantizedDotArguments(
        query, qWeight, byteOffset, dimensions, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    return IMPL.ggufQ8_0DotProduct(query, qWeight, byteOffset, dimensions);
  }

  /**
   * Dot product of a full-precision query with one GGUF Q6_K quantized row.
   *
   * <p>The operation fuses Q6_K dequantization and dot-product accumulation without allocating a
   * temporary decoded row.
   */
  public static float ggufQ6_KDotProduct(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions) {
    checkGgufQuantizedDotArguments(
        query,
        qWeight,
        byteOffset,
        dimensions,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    return IMPL.ggufQ6_KDotProduct(query, qWeight, byteOffset, dimensions);
  }

  /** Dequantizes GGUF Q6_K blocks into a caller-owned float array. */
  public static void ggufQ6_KDequantize(
      MemorySegment qWeight, long byteOffset, float[] out, int outOffset, int dimensions) {
    Objects.requireNonNull(qWeight, "qWeight");
    Objects.requireNonNull(out, "out");
    if (byteOffset < 0) {
      throw new IllegalArgumentException("byteOffset must be >= 0: " + byteOffset);
    }
    if (outOffset < 0 || dimensions < 0 || outOffset > out.length - dimensions) {
      throw new IllegalArgumentException(
          "out range is invalid: offset="
              + outOffset
              + ", dimensions="
              + dimensions
              + ", length="
              + out.length);
    }
    if (dimensions % VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "GGUF Q6_K dimensions must be a multiple of 256: " + dimensions);
    }
    long required =
        byteOffset
            + ggufQuantizedRowBytes(
                dimensions,
                VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE,
                VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    if (required < byteOffset || required > qWeight.byteSize()) {
      throw new IllegalArgumentException(
          "qWeight byteSize is too small for requested values: "
              + qWeight.byteSize()
              + " < "
              + required);
    }
    IMPL.ggufQ6_KDequantize(qWeight, byteOffset, out, outOffset, dimensions);
  }

  /** Batched row-major GEMV over GGUF Q4_0 rows. */
  public static void ggufQ4_0BatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    IMPL.ggufQ4_0MatVecDot(query, qWeight, rows, cols, out);
  }

  /**
   * GGML-compatible GEMV over Q4_0 rows using a Q8_0-quantized activation vector.
   *
   * <p>The query is quantized once per call into caller-owned scratch arrays, then reused for every
   * matrix row.
   *
   * @param q8Quants scratch space with at least {@code cols} entries
   * @param q8Scales scratch space with at least {@code cols / 32} entries
   * @param q8ZeroPointCorrections scratch space with at least {@code cols / 4} entries; kernels
   *     using unsigned Q4 arithmetic populate it during activation quantization
   */
  public static void ggufQ4_0Q8_0BatchDotProduct(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections) {
    ggufQ4_0Q8_0BatchDotProduct(
        query,
        qWeight,
        rows,
        cols,
        out,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        GgufQ4Kernel.WIDENED);
  }

  /** Q4_0 by Q8_0 GEMV with an explicit arithmetic-kernel policy. */
  public static void ggufQ4_0Q8_0BatchDotProduct(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      GgufQ4Kernel kernel) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    checkGgufQ4CorrectionScratch(q8ZeroPointCorrections, cols);
    IMPL.ggufQ4_0Q8_0MatVecDot(
        query,
        qWeight,
        rows,
        cols,
        out,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        Objects.requireNonNull(kernel, "kernel"));
  }

  /**
   * Multiplies two Q4_0 matrices by the same activation with one Q8_0 quantization and one row
   * dispatch.
   *
   * <p>This is intended for transformer projections such as SwiGLU gate/up matrices that consume
   * the same normalized activation.
   */
  public static void ggufQ4_0Q8_0DualBatchDotProduct(
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
      int[] q8ZeroPointCorrections) {
    ggufQ4_0Q8_0DualBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        GgufQ4Kernel.WIDENED);
  }

  /** Two Q4_0 projections with an explicit arithmetic-kernel policy. */
  public static void ggufQ4_0Q8_0DualBatchDotProduct(
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
      int[] q8ZeroPointCorrections,
      GgufQ4Kernel kernel) {
    checkGgufQuantizedBatchArguments(
        query, firstWeight, firstRows, cols, firstOut, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query, secondWeight, secondRows, cols, secondOut, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    checkGgufQ4CorrectionScratch(q8ZeroPointCorrections, cols);
    IMPL.ggufQ4_0Q8_0DualMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        Objects.requireNonNull(kernel, "kernel"));
  }

  /**
   * Multiplies three Q4_0 matrices by the same activation with one Q8_0 quantization and one row
   * dispatch.
   *
   * <p>This is intended for grouped query/key/value transformer projections.
   */
  public static void ggufQ4_0Q8_0TripleBatchDotProduct(
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
      int[] q8ZeroPointCorrections) {
    ggufQ4_0Q8_0TripleBatchDotProduct(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        GgufQ4Kernel.WIDENED);
  }

  /** Three Q4_0 projections with an explicit arithmetic-kernel policy. */
  public static void ggufQ4_0Q8_0TripleBatchDotProduct(
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
      int[] q8ZeroPointCorrections,
      GgufQ4Kernel kernel) {
    checkGgufQuantizedBatchArguments(
        query, firstWeight, firstRows, cols, firstOut, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query, secondWeight, secondRows, cols, secondOut, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query, thirdWeight, thirdRows, cols, thirdOut, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    checkGgufQ4CorrectionScratch(q8ZeroPointCorrections, cols);
    IMPL.ggufQ4_0Q8_0TripleMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        Objects.requireNonNull(kernel, "kernel"));
  }

  /**
   * Multiplies one Q4_0 matrix by a batch-major collection of activation vectors.
   *
   * <p>{@code queries} is laid out as {@code [batchSize][cols]} and {@code out} as {@code
   * [batchSize][rows]}. Activation quantization scratch is caller-owned so repeated prefill calls
   * do not allocate in the hot path.
   *
   * @param q8Quants scratch space with at least {@code batchSize * cols} entries
   * @param q8Scales scratch space with at least {@code batchSize * (cols / 32)} entries
   * @param q8ZeroPointCorrections scratch space with at least {@code batchSize * cols / 4} entries;
   *     kernels using unsigned Q4 arithmetic populate it during activation quantization
   */
  public static void ggufQ4_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections) {
    ggufQ4_0Q8_0BatchedMatmul(
        queries,
        qWeight,
        batchSize,
        rows,
        cols,
        out,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        GgufQ4Kernel.WIDENED);
  }

  /** Q4_0 batched matrix multiplication with an explicit arithmetic-kernel policy. */
  public static void ggufQ4_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      GgufQ4Kernel kernel) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be >= 1: " + rows);
    }
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    ggufQ4_0Q8_0BatchedMatmul(
        queries,
        qWeight,
        batchSize,
        rows,
        cols,
        out,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        new float[checkedProduct(outputEntries, 8, "Q4 lane scratch")],
        kernel);
  }

  /**
   * Allocation-free Q4_0 batched matrix multiplication with caller-owned reduction lanes.
   *
   * @param q8ZeroPointCorrections scratch space with at least {@code batchSize * cols / 4} entries
   * @param laneScratch scratch space with at least {@code batchSize * rows * 8} entries
   */
  public static void ggufQ4_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch) {
    ggufQ4_0Q8_0BatchedMatmul(
        queries,
        qWeight,
        batchSize,
        rows,
        cols,
        out,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        GgufQ4Kernel.WIDENED);
  }

  /** Allocation-free Q4_0 batched matrix multiplication with an explicit kernel policy. */
  public static void ggufQ4_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(q8Quants, "q8Quants");
    Objects.requireNonNull(q8Scales, "q8Scales");
    Objects.requireNonNull(q8ZeroPointCorrections, "q8ZeroPointCorrections");
    Objects.requireNonNull(laneScratch, "laneScratch");
    checkGgufQuantizedMatrixArguments(
        qWeight,
        rows,
        cols,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    int queryEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    int laneEntries = checkedProduct(outputEntries, 8, "Q4 lane scratch");
    int scaleEntries =
        checkedProduct(batchSize, cols / VectorUtilSupport.GGUF_Q_BLOCK_SIZE, "batch scales");
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries.length must be >= batchSize * cols: " + queries.length + " < " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
    if (q8Quants.length < queryEntries) {
      throw new IllegalArgumentException(
          "q8Quants.length must be >= batchSize * cols: " + q8Quants.length + " < " + queryEntries);
    }
    if (q8Scales.length < scaleEntries) {
      throw new IllegalArgumentException(
          "q8Scales.length must be >= batch scales: " + q8Scales.length + " < " + scaleEntries);
    }
    checkGgufQ4CorrectionScratch(q8ZeroPointCorrections, queryEntries);
    if (laneScratch.length < laneEntries) {
      throw new IllegalArgumentException(
          "lane scratch length must be >= batchSize * rows * 8: "
              + laneScratch.length
              + " < "
              + laneEntries);
    }
    IMPL.ggufQ4_0Q8_0BatchedMatmul(
        queries,
        qWeight,
        batchSize,
        rows,
        cols,
        out,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        Objects.requireNonNull(kernel, "kernel"));
  }

  /**
   * Computes a non-empty Q4_0 output-row range from a caller-prequantized activation batch.
   *
   * <p>This low-level operation performs no quantization and no worker publication, allowing a
   * {@link GgufStagePlan} to compose dependent matrix stages without nested scheduling.
   */
  public static void ggufQ4_0Q8_0BatchedMatmulRows(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      int fromRow,
      int toRow,
      float[] out,
      GgufQ8_0Batch activation,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    ggufQ4_0Q8_0BatchedMatmulRows(
        qWeight, batchSize, rows, cols, fromRow, toRow, out, activation, laneScratch, 0, kernel);
  }

  /**
   * Computes a Q4_0 output-row range using an interior row window of shared lane scratch.
   *
   * <p>Distinct matrix ranges may execute concurrently when their scratch-row windows do not
   * overlap.
   */
  public static void ggufQ4_0Q8_0BatchedMatmulRows(
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      int fromRow,
      int toRow,
      float[] out,
      GgufQ8_0Batch activation,
      float[] laneScratch,
      int laneScratchRowOffset,
      GgufQ4Kernel kernel) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be >= 1: " + rows);
    }
    if (fromRow < 0 || fromRow >= toRow || toRow > rows) {
      throw new IndexOutOfBoundsException(
          "row range must satisfy 0 <= fromRow < toRow <= rows: "
              + fromRow
              + ".."
              + toRow
              + " for "
              + rows);
    }
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(activation, "activation");
    Objects.requireNonNull(laneScratch, "laneScratch");
    Objects.requireNonNull(kernel, "kernel");
    if (laneScratchRowOffset < 0) {
      throw new IllegalArgumentException(
          "laneScratchRowOffset must be non-negative: " + laneScratchRowOffset);
    }
    checkGgufQuantizedMatrixArguments(
        qWeight,
        rows,
        cols,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    if (batchSize > activation.batchCapacity()) {
      throw new IllegalArgumentException(
          "batchSize exceeds activation capacity: "
              + batchSize
              + " > "
              + activation.batchCapacity());
    }
    if (cols != activation.dimensions()) {
      throw new IllegalArgumentException(
          "cols must equal activation dimensions: " + cols + " != " + activation.dimensions());
    }
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
    checkGgufQ4LaneScratch(
        laneScratch,
        batchSize,
        Math.addExact(laneScratchRowOffset, rows),
        "Q4 row-range lane scratch");
    IMPL.ggufQ4_0Q8_0BatchedMatmulRows(
        qWeight,
        batchSize,
        rows,
        cols,
        fromRow,
        toRow,
        out,
        activation,
        laneScratch,
        laneScratchRowOffset,
        kernel);
  }

  /** Two Q4_0 matrices over an activation batch with one Q8_0 quantization and row dispatch. */
  public static void ggufQ4_0Q8_0DualBatchedMatmul(
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
      int[] q8ZeroPointCorrections,
      float[] laneScratch) {
    ggufQ4_0Q8_0DualBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        GgufQ4Kernel.WIDENED);
  }

  /** Two Q4_0 batched projections with an explicit arithmetic-kernel policy. */
  public static void ggufQ4_0Q8_0DualBatchedMatmul(
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
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    checkGgufQuantizedBatchedArguments(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    int activationEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    checkGgufActivationScratch(
        q8Quants, q8Scales, activationEntries, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    checkGgufQ4CorrectionScratch(q8ZeroPointCorrections, activationEntries);
    checkGgufQ4LaneScratch(
        laneScratch, batchSize, Math.addExact(firstRows, secondRows), "dual Q4 lane scratch");
    IMPL.ggufQ4_0Q8_0DualBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        Objects.requireNonNull(kernel, "kernel"));
  }

  /** Three Q4_0 matrices over an activation batch with one Q8_0 quantization and row dispatch. */
  public static void ggufQ4_0Q8_0TripleBatchedMatmul(
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
      int[] q8ZeroPointCorrections,
      float[] laneScratch) {
    ggufQ4_0Q8_0TripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        GgufQ4Kernel.WIDENED);
  }

  /** Three Q4_0 batched projections with an explicit arithmetic-kernel policy. */
  public static void ggufQ4_0Q8_0TripleBatchedMatmul(
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
      int[] q8ZeroPointCorrections,
      float[] laneScratch,
      GgufQ4Kernel kernel) {
    checkGgufQuantizedBatchedArguments(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        thirdWeight,
        batchSize,
        thirdRows,
        cols,
        thirdOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    int activationEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    checkGgufActivationScratch(
        q8Quants, q8Scales, activationEntries, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    checkGgufQ4CorrectionScratch(q8ZeroPointCorrections, activationEntries);
    int totalRows = Math.addExact(Math.addExact(firstRows, secondRows), thirdRows);
    checkGgufQ4LaneScratch(laneScratch, batchSize, totalRows, "triple Q4 lane scratch");
    IMPL.ggufQ4_0Q8_0TripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8ZeroPointCorrections,
        laneScratch,
        Objects.requireNonNull(kernel, "kernel"));
  }

  /** Batched row-major GEMV over GGUF Q4_K rows. */
  public static void ggufQ4_KBatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query,
        qWeight,
        rows,
        cols,
        out,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    IMPL.ggufQ4_KMatVecDot(query, qWeight, rows, cols, out);
  }

  /**
   * GGML-compatible GEMV over Q4_K rows using a Q8_K-quantized activation vector.
   *
   * <p>The query is quantized once per call into caller-owned scratch arrays, then reused for every
   * matrix row. The 16-value sums carry Q8_K's block sums used by Q4_K's minimum correction.
   *
   * @param q8Quants scratch space with at least {@code cols} entries
   * @param q8Scales scratch space with at least {@code cols / 256} entries
   * @param q8Sums scratch space with at least {@code cols / 16} entries
   */
  public static void ggufQ4_KQ8_KBatchDotProduct(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    checkGgufQuantizedBatchArguments(
        query,
        qWeight,
        rows,
        cols,
        out,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, cols);
    IMPL.ggufQ4_KQ8_KMatVecDot(query, qWeight, rows, cols, out, q8Quants, q8Scales, q8Sums);
  }

  /**
   * Multiplies one Q4_K matrix by a batch-major collection of activation vectors.
   *
   * <p>{@code queries} is laid out as {@code [batchSize][cols]} and {@code out} as {@code
   * [batchSize][rows]}. Each activation row is quantized independently to Q8_K in caller-owned
   * scratch before the matrix rows are evaluated.
   *
   * @param q8Quants scratch space with at least {@code batchSize * cols} entries
   * @param q8Scales scratch space with at least {@code batchSize * (cols / 256)} entries
   * @param q8Sums scratch space with at least {@code batchSize * (cols / 16)} entries
   */
  public static void ggufQ4_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be >= 1: " + rows);
    }
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(q8Quants, "q8Quants");
    Objects.requireNonNull(q8Scales, "q8Scales");
    Objects.requireNonNull(q8Sums, "q8Sums");
    checkGgufQuantizedMatrixArguments(
        qWeight,
        rows,
        cols,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    int queryEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    int scaleEntries =
        checkedProduct(
            batchSize, cols / VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE, "batch Q8_K scales");
    int sumEntries =
        checkedProduct(
            batchSize, cols / VectorUtilSupport.GGUF_Q8_K_SUM_BLOCK_SIZE, "batch Q8_K sums");
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries.length must be >= batchSize * cols: " + queries.length + " < " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
    if (q8Quants.length < queryEntries) {
      throw new IllegalArgumentException(
          "q8Quants.length must be >= batchSize * cols: " + q8Quants.length + " < " + queryEntries);
    }
    if (q8Scales.length < scaleEntries) {
      throw new IllegalArgumentException(
          "q8Scales.length must be >= batch Q8_K scales: "
              + q8Scales.length
              + " < "
              + scaleEntries);
    }
    if (q8Sums.length < sumEntries) {
      throw new IllegalArgumentException(
          "q8Sums.length must be >= batch Q8_K sums: " + q8Sums.length + " < " + sumEntries);
    }
    IMPL.ggufQ4_KQ8_KBatchedMatmul(
        queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales, q8Sums);
  }

  /**
   * Multiplies two Q4_K matrices by a batch of activations with one Q8_K quantization pass and one
   * row dispatch.
   */
  public static void ggufQ4_KQ8_KDualBatchedMatmul(
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
    checkGgufQuantizedBatchedArguments(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    int activationEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    checkGgufActivationScratch(
        q8Quants, q8Scales, activationEntries, VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, activationEntries);
    IMPL.ggufQ4_KQ8_KDualBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  /** Multiplies two Q4_K matrices by one shared Q8_K activation quantization and row dispatch. */
  public static void ggufQ4_KQ8_KDualBatchDotProduct(
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
    checkGgufQuantizedBatchArguments(
        query,
        firstWeight,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        secondWeight,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, cols);
    IMPL.ggufQ4_KQ8_KDualMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  /** Multiplies three Q4_K matrices by one shared Q8_K activation quantization and row dispatch. */
  public static void ggufQ4_KQ8_KTripleBatchDotProduct(
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
    checkGgufQuantizedBatchArguments(
        query,
        firstWeight,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        secondWeight,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        thirdWeight,
        thirdRows,
        cols,
        thirdOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, cols);
    IMPL.ggufQ4_KQ8_KTripleMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  /**
   * Multiplies two Q4_K matrices and one Q6_K matrix by one shared Q8_K activation quantization and
   * row dispatch.
   */
  public static void ggufQ4_KQ4_KQ6_KQ8_KTripleBatchDotProduct(
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
    checkGgufQuantizedBatchArguments(
        query,
        firstWeight,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        secondWeight,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        thirdWeight,
        thirdRows,
        cols,
        thirdOut,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, cols);
    IMPL.ggufQ4_KQ4_KQ6_KQ8_KTripleMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  /**
   * Multiplies two Q4_K matrices and one Q6_K matrix by a batch of activations with one Q8_K
   * quantization pass and one row dispatch.
   */
  public static void ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
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
    checkGgufQuantizedBatchedArguments(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q4_K_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        thirdWeight,
        batchSize,
        thirdRows,
        cols,
        thirdOut,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    int activationEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    checkGgufActivationScratch(
        q8Quants, q8Scales, activationEntries, VectorUtilSupport.GGUF_Q4_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, activationEntries);
    IMPL.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  /** Batched row-major GEMV over GGUF Q5_K rows. */
  public static void ggufQ5_KBatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query,
        qWeight,
        rows,
        cols,
        out,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    IMPL.ggufQ5_KMatVecDot(query, qWeight, rows, cols, out);
  }

  /**
   * GGML-compatible GEMV over Q5_K rows using a Q8_K-quantized activation vector.
   *
   * @param q8Quants scratch space with at least {@code cols} entries
   * @param q8Scales scratch space with at least {@code cols / 256} entries
   * @param q8Sums scratch space with at least {@code cols / 16} entries
   */
  public static void ggufQ5_KQ8_KBatchDotProduct(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    checkGgufQuantizedBatchArguments(
        query,
        qWeight,
        rows,
        cols,
        out,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, cols);
    IMPL.ggufQ5_KQ8_KMatVecDot(query, qWeight, rows, cols, out, q8Quants, q8Scales, q8Sums);
  }

  /**
   * Multiplies one Q5_K matrix by a batch-major collection of activation vectors.
   *
   * <p>{@code queries} is laid out as {@code [batchSize][cols]} and {@code out} as {@code
   * [batchSize][rows]}. Each activation row is quantized independently to Q8_K in caller-owned
   * scratch before the matrix rows are evaluated.
   *
   * @param q8Quants scratch space with at least {@code batchSize * cols} entries
   * @param q8Scales scratch space with at least {@code batchSize * (cols / 256)} entries
   * @param q8Sums scratch space with at least {@code batchSize * (cols / 16)} entries
   */
  public static void ggufQ5_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales,
      short[] q8Sums) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be >= 1: " + rows);
    }
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(q8Quants, "q8Quants");
    Objects.requireNonNull(q8Scales, "q8Scales");
    Objects.requireNonNull(q8Sums, "q8Sums");
    checkGgufQuantizedMatrixArguments(
        qWeight,
        rows,
        cols,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    int queryEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    int scaleEntries =
        checkedProduct(
            batchSize, cols / VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE, "batch Q8_K scales");
    int sumEntries =
        checkedProduct(
            batchSize, cols / VectorUtilSupport.GGUF_Q8_K_SUM_BLOCK_SIZE, "batch Q8_K sums");
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries.length must be >= batchSize * cols: " + queries.length + " < " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
    if (q8Quants.length < queryEntries) {
      throw new IllegalArgumentException(
          "q8Quants.length must be >= batchSize * cols: " + q8Quants.length + " < " + queryEntries);
    }
    if (q8Scales.length < scaleEntries) {
      throw new IllegalArgumentException(
          "q8Scales.length must be >= batch Q8_K scales: "
              + q8Scales.length
              + " < "
              + scaleEntries);
    }
    if (q8Sums.length < sumEntries) {
      throw new IllegalArgumentException(
          "q8Sums.length must be >= batch Q8_K sums: " + q8Sums.length + " < " + sumEntries);
    }
    IMPL.ggufQ5_KQ8_KBatchedMatmul(
        queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales, q8Sums);
  }

  /** Multiplies two Q5_K matrices by one shared Q8_K activation quantization and row dispatch. */
  public static void ggufQ5_KQ8_KDualBatchDotProduct(
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
    checkGgufQuantizedBatchArguments(
        query,
        firstWeight,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        secondWeight,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, cols);
    IMPL.ggufQ5_KQ8_KDualMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  /** Multiplies three Q5_K matrices by one shared Q8_K activation quantization and row dispatch. */
  public static void ggufQ5_KQ8_KTripleBatchDotProduct(
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
    checkGgufQuantizedBatchArguments(
        query,
        firstWeight,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        secondWeight,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query,
        thirdWeight,
        thirdRows,
        cols,
        thirdOut,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q5_K_BLOCK_SIZE);
    checkGgufQ8KSums(q8Sums, cols);
    IMPL.ggufQ5_KQ8_KTripleMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales,
        q8Sums);
  }

  /** Batched row-major GEMV over GGUF Q5_0 rows. */
  public static void ggufQ5_0BatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q5_0_BLOCK_BYTES);
    IMPL.ggufQ5_0MatVecDot(query, qWeight, rows, cols, out);
  }

  /**
   * GGML-compatible GEMV over Q5_0 rows using a Q8_0-quantized activation vector.
   *
   * @param q8Quants scratch space with at least {@code cols} entries
   * @param q8Scales scratch space with at least {@code cols / 32} entries
   */
  public static void ggufQ5_0Q8_0BatchDotProduct(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q5_0_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    IMPL.ggufQ5_0Q8_0MatVecDot(query, qWeight, rows, cols, out, q8Quants, q8Scales);
  }

  /**
   * Multiplies one Q5_0 matrix by a batch-major collection of activation vectors.
   *
   * <p>{@code queries} is laid out as {@code [batchSize][cols]} and {@code out} as {@code
   * [batchSize][rows]}. Each activation row is quantized independently to Q8_0 in caller-owned
   * scratch before the matrix rows are evaluated.
   *
   * @param q8Quants scratch space with at least {@code batchSize * cols} entries
   * @param q8Scales scratch space with at least {@code batchSize * (cols / 32)} entries
   */
  public static void ggufQ5_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be >= 1: " + rows);
    }
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(q8Quants, "q8Quants");
    Objects.requireNonNull(q8Scales, "q8Scales");
    checkGgufQuantizedMatrixArguments(
        qWeight,
        rows,
        cols,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q5_0_BLOCK_BYTES);
    int queryEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    int scaleEntries =
        checkedProduct(batchSize, cols / VectorUtilSupport.GGUF_Q_BLOCK_SIZE, "batch Q8_0 scales");
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries.length must be >= batchSize * cols: " + queries.length + " < " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
    if (q8Quants.length < queryEntries) {
      throw new IllegalArgumentException(
          "q8Quants.length must be >= batchSize * cols: " + q8Quants.length + " < " + queryEntries);
    }
    if (q8Scales.length < scaleEntries) {
      throw new IllegalArgumentException(
          "q8Scales.length must be >= batch Q8_0 scales: "
              + q8Scales.length
              + " < "
              + scaleEntries);
    }
    IMPL.ggufQ5_0Q8_0BatchedMatmul(
        queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales);
  }

  /** Batched row-major GEMV over GGUF Q8_0 rows. */
  public static void ggufQ8_0BatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    IMPL.ggufQ8_0MatVecDot(query, qWeight, rows, cols, out);
  }

  /**
   * GGML-compatible GEMV over Q8_0 rows using a Q8_0-quantized activation vector.
   *
   * <p>The query is quantized once per call into caller-owned scratch arrays, then reused for every
   * matrix row.
   *
   * @param q8Quants scratch space with at least {@code cols} entries
   * @param q8Scales scratch space with at least {@code cols / 32} entries
   */
  public static void ggufQ8_0Q8_0BatchDotProduct(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    IMPL.ggufQ8_0Q8_0MatVecDot(query, qWeight, rows, cols, out, q8Quants, q8Scales);
  }

  /**
   * Multiplies one Q8_0 matrix by a batch-major collection of activation vectors.
   *
   * <p>{@code queries} is laid out as {@code [batchSize][cols]} and {@code out} as {@code
   * [batchSize][rows]}. Each activation row is quantized independently to Q8_0 in caller-owned
   * scratch before the matrix rows are evaluated.
   *
   * @param q8Quants scratch space with at least {@code batchSize * cols} entries
   * @param q8Scales scratch space with at least {@code batchSize * (cols / 32)} entries
   */
  public static void ggufQ8_0Q8_0BatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be >= 1: " + rows);
    }
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(q8Quants, "q8Quants");
    Objects.requireNonNull(q8Scales, "q8Scales");
    checkGgufQuantizedMatrixArguments(
        qWeight,
        rows,
        cols,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    int queryEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    int scaleEntries =
        checkedProduct(batchSize, cols / VectorUtilSupport.GGUF_Q_BLOCK_SIZE, "batch Q8_0 scales");
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries.length must be >= batchSize * cols: " + queries.length + " < " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
    if (q8Quants.length < queryEntries) {
      throw new IllegalArgumentException(
          "q8Quants.length must be >= batchSize * cols: " + q8Quants.length + " < " + queryEntries);
    }
    if (q8Scales.length < scaleEntries) {
      throw new IllegalArgumentException(
          "q8Scales.length must be >= batch Q8_0 scales: "
              + q8Scales.length
              + " < "
              + scaleEntries);
    }
    IMPL.ggufQ8_0Q8_0BatchedMatmul(
        queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales);
  }

  /** Two Q8_0 matrices over an activation batch with one Q8_0 quantization and row dispatch. */
  public static void ggufQ8_0Q8_0DualBatchedMatmul(
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
      float[] q8Scales) {
    checkGgufQuantizedBatchedArguments(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    int activationEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    checkGgufActivationScratch(
        q8Quants, q8Scales, activationEntries, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    IMPL.ggufQ8_0Q8_0DualBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales);
  }

  /** Three Q8_0 matrices over an activation batch with one Q8_0 quantization and row dispatch. */
  public static void ggufQ8_0Q8_0TripleBatchedMatmul(
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
      float[] q8Scales) {
    checkGgufQuantizedBatchedArguments(
        queries,
        firstWeight,
        batchSize,
        firstRows,
        cols,
        firstOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        secondWeight,
        batchSize,
        secondRows,
        cols,
        secondOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufQuantizedBatchedArguments(
        queries,
        thirdWeight,
        batchSize,
        thirdRows,
        cols,
        thirdOut,
        VectorUtilSupport.GGUF_Q_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    int activationEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    checkGgufActivationScratch(
        q8Quants, q8Scales, activationEntries, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    IMPL.ggufQ8_0Q8_0TripleBatchedMatmul(
        queries,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        batchSize,
        cols,
        q8Quants,
        q8Scales);
  }

  /**
   * Multiplies two Q8_0 matrices by the same activation with one Q8_0 quantization and one row
   * dispatch.
   */
  public static void ggufQ8_0Q8_0DualBatchDotProduct(
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
    checkGgufQuantizedBatchArguments(
        query, firstWeight, firstRows, cols, firstOut, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query, secondWeight, secondRows, cols, secondOut, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    IMPL.ggufQ8_0Q8_0DualMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        cols,
        q8Quants,
        q8Scales);
  }

  /**
   * Multiplies three Q8_0 matrices by the same activation with one Q8_0 quantization and one row
   * dispatch.
   */
  public static void ggufQ8_0Q8_0TripleBatchDotProduct(
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
    checkGgufQuantizedBatchArguments(
        query, firstWeight, firstRows, cols, firstOut, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query, secondWeight, secondRows, cols, secondOut, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufQuantizedBatchArguments(
        query, thirdWeight, thirdRows, cols, thirdOut, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
    IMPL.ggufQ8_0Q8_0TripleMatVecDot(
        query,
        firstWeight,
        firstRows,
        firstOut,
        secondWeight,
        secondRows,
        secondOut,
        thirdWeight,
        thirdRows,
        thirdOut,
        cols,
        q8Quants,
        q8Scales);
  }

  /** Batched row-major GEMV over GGUF Q6_K rows. */
  public static void ggufQ6_KBatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query,
        qWeight,
        rows,
        cols,
        out,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    IMPL.ggufQ6_KMatVecDot(query, qWeight, rows, cols, out);
  }

  /**
   * GGML-compatible GEMV over Q6_K rows using a Q8_K-quantized activation vector.
   *
   * <p>The query is quantized once per call into caller-owned scratch arrays, then reused for every
   * matrix row. This matches GGML's Q6_K matrix multiplication semantics while keeping the hot path
   * free of large temporary allocations.
   *
   * @param q8Quants scratch space with at least {@code cols} entries
   * @param q8Scales scratch space with at least {@code cols / 256} entries
   */
  public static void ggufQ6_KQ8_KBatchDotProduct(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    checkGgufQuantizedBatchArguments(
        query,
        qWeight,
        rows,
        cols,
        out,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    checkGgufActivationScratch(q8Quants, q8Scales, cols, VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE);
    IMPL.ggufQ6_KQ8_KMatVecDot(query, qWeight, rows, cols, out, q8Quants, q8Scales);
  }

  /**
   * Multiplies one Q6_K matrix by a batch-major collection of activation vectors.
   *
   * <p>{@code queries} is laid out as {@code [batchSize][cols]} and {@code out} as {@code
   * [batchSize][rows]}. Each activation row is quantized independently to Q8_K in caller-owned
   * scratch before the matrix rows are evaluated.
   *
   * @param q8Quants scratch space with at least {@code batchSize * cols} entries
   * @param q8Scales scratch space with at least {@code batchSize * (cols / 256)} entries
   */
  public static void ggufQ6_KQ8_KBatchedMatmul(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      byte[] q8Quants,
      float[] q8Scales) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be >= 1: " + rows);
    }
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(q8Quants, "q8Quants");
    Objects.requireNonNull(q8Scales, "q8Scales");
    checkGgufQuantizedMatrixArguments(
        qWeight,
        rows,
        cols,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE,
        VectorUtilSupport.GGUF_Q6_K_BLOCK_BYTES);
    int queryEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    int scaleEntries =
        checkedProduct(
            batchSize, cols / VectorUtilSupport.GGUF_Q6_K_BLOCK_SIZE, "batch Q8_K scales");
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries.length must be >= batchSize * cols: " + queries.length + " < " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
    if (q8Quants.length < queryEntries) {
      throw new IllegalArgumentException(
          "q8Quants.length must be >= batchSize * cols: " + q8Quants.length + " < " + queryEntries);
    }
    if (q8Scales.length < scaleEntries) {
      throw new IllegalArgumentException(
          "q8Scales.length must be >= batch Q8_K scales: "
              + q8Scales.length
              + " < "
              + scaleEntries);
    }
    IMPL.ggufQ6_KQ8_KBatchedMatmul(
        queries, qWeight, batchSize, rows, cols, out, q8Quants, q8Scales);
  }

  /**
   * Fills {@code out[i]} with the squared L2 distance from {@code query} to {@code matrix[i]} for
   * all rows {@code i} in {@code [0, matrix.length)}. {@code out.length} must be &ge; {@code
   * matrix.length}.
   *
   * <p>Delegates to {@link VectorUtilSupport#matVecSquaredL2} with a fused 4-row SIMD kernel.
   */
  public static void batchSquaredL2(float[] query, float[][] matrix, float[] out) {
    checkBatchArguments(query, matrix, out, matrix.length);
    IMPL.matVecSquaredL2(query, matrix, out, matrix.length);
  }

  /**
   * Fused GEMV squared-L2 over the first {@code numRows} rows of {@code matrix}. Useful when {@code
   * matrix} is a reusable scratch buffer sized for a maximum batch but the current call processes
   * only a prefix.
   */
  public static void batchSquaredL2(float[] query, float[][] matrix, float[] out, int numRows) {
    checkBatchArguments(query, matrix, out, numRows);
    IMPL.matVecSquaredL2(query, matrix, out, numRows);
  }

  /**
   * Off-heap fused GEMV dot product: fills {@code out[i]} with the dot product of {@code query} and
   * the off-heap row {@code rows[i]} for {@code i in [0, count)}. Each {@code rows[i]} is a {@link
   * MemorySegment} holding {@code dim} little-endian float32s (typically a zero-copy mmap slice).
   *
   * <p>Delegates to {@link VectorUtilSupport#matVecDot(float[], MemorySegment[], int, float[],
   * int)}, which SIMD implementations override with a fused 4-row kernel that loads each query SIMD
   * chunk once and applies it to 4 segment rows simultaneously — the zero-copy analogue of {@link
   * #batchDotProduct(float[], float[][], float[], int)}.
   */
  public static void batchDotProduct(
      float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    checkBatchSegmentArguments(query, rows, dim, out, count);
    IMPL.matVecDot(query, rows, dim, out, count);
  }

  /**
   * Off-heap fused GEMV squared-L2: fills {@code out[i]} with the squared L2 distance from {@code
   * query} to the off-heap row {@code rows[i]} for {@code i in [0, count)}. Zero-copy analogue of
   * {@link #batchSquaredL2(float[], float[][], float[], int)}.
   */
  public static void batchSquaredL2(
      float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    checkBatchSegmentArguments(query, rows, dim, out, count);
    IMPL.matVecSquaredL2(query, rows, dim, out, count);
  }

  /**
   * Fills {@code out[i]} with the raw cosine similarity (in {@code [-1, 1]}) of {@code query} and
   * {@code rows[i]} for {@code i in [0, count)}. Callers apply the {@code (1+cos)/2} score
   * transform.
   *
   * <p>Delegates to {@link VectorUtilSupport#batchCosine(float[], float[][], float[], int)}, which
   * SIMD implementations override with a fused 4-row kernel that computes the query norm once and
   * loads each query SIMD chunk once for 4 rows — the cosine analogue of {@link
   * #batchDotProduct(float[], float[][], float[], int)}.
   */
  public static void batchCosine(float[] query, float[][] rows, float[] out, int count) {
    checkBatchArguments(query, rows, out, count);
    IMPL.batchCosine(query, rows, out, count);
  }

  /**
   * Off-heap fused GEMV cosine: fills {@code out[i]} with the raw cosine similarity of {@code
   * query} and the off-heap row {@code rows[i]} for {@code i in [0, count)}. Zero-copy analogue of
   * {@link #batchCosine(float[], float[][], float[], int)}.
   */
  public static void batchCosine(
      float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    checkBatchSegmentArguments(query, rows, dim, out, count);
    IMPL.batchCosine(query, rows, dim, out, count);
  }

  // --- PQ ADC (Asymmetric Distance Computation) kernels ---

  /**
   * Sums one entry per subspace from {@code table}, indexed by the unsigned PQ codes starting at
   * {@code codesOffset}. Returns the raw partial sum — callers typically remap it through a
   * similarity-specific transform (e.g. {@code 1/(1+d)} for L2).
   *
   * @see VectorUtilSupport#assembleAndSum
   */
  public static float assembleAndSum(
      float[][] table, byte[] codes, int codesOffset, int numSubspaces) {
    return IMPL.assembleAndSum(table, codes, codesOffset, numSubspaces);
  }

  /**
   * Batched ADC scorer: fills {@code out[i]} with the partial-sum score for the {@code i}-th
   * neighbor whose M codes occupy bytes {@code [codesOffset + i*M, codesOffset + (i+1)*M)} of
   * {@code packedCodes}. See {@link VectorUtilSupport#batchAssembleAndSum} for the 4-row unroll
   * pattern.
   */
  public static void batchAssembleAndSum(
      float[][] table,
      byte[] packedCodes,
      int codesOffset,
      float[] out,
      int count,
      int numSubspaces) {
    IMPL.batchAssembleAndSum(table, packedCodes, codesOffset, out, count, numSubspaces);
  }

  // --- Normalization ---

  /**
   * L2-normalizes the given vector in place and returns it.
   *
   * @param v the vector to normalize
   * @param throwOnZero if true, throws for zero-length vectors
   * @return the normalized vector (same array reference)
   */
  public static float[] l2normalize(float[] v, boolean throwOnZero) {
    return IMPL.l2normalize(v, throwOnZero);
  }

  private static void checkDimensions(int len1, int len2) {
    if (len1 != len2) {
      throw new IllegalArgumentException("Vector dimensions differ: " + len1 + " != " + len2);
    }
  }

  private static void checkBatchArguments(
      float[] query, float[][] matrix, float[] out, int numRows) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(matrix, "matrix");
    Objects.requireNonNull(out, "out");
    if (numRows < 0 || numRows > matrix.length) {
      throw new IllegalArgumentException(
          "numRows must be in [0, matrix.length]: " + numRows + " for " + matrix.length);
    }
    if (out.length < numRows) {
      throw new IllegalArgumentException(
          "out.length must be >= numRows: " + out.length + " < " + numRows);
    }
    // Lazy failure messages: building "matrix[" + i + "]" eagerly per row (Objects.requireNonNull
    // evaluates its message arg unconditionally) cost ~18% of search time in a JFR profile — the
    // string concatenation ran on every batch-score call. Only construct the message on failure.
    int qlen = query.length;
    for (int i = 0; i < numRows; i++) {
      float[] row = matrix[i];
      if (row == null) {
        throw new NullPointerException("matrix[" + i + "]");
      }
      if (row.length != qlen) {
        throw new IllegalArgumentException(
            "Vector dimensions differ at row " + i + ": " + qlen + " != " + row.length);
      }
    }
  }

  private static void checkBatchSegmentArguments(
      float[] query, MemorySegment[] rows, int dim, float[] out, int count) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(rows, "rows");
    Objects.requireNonNull(out, "out");
    checkDimensions(query.length, dim);
    if (count < 0 || count > rows.length) {
      throw new IllegalArgumentException(
          "count must be in [0, rows.length]: " + count + " for " + rows.length);
    }
    if (out.length < count) {
      throw new IllegalArgumentException(
          "out.length must be >= count: " + out.length + " < " + count);
    }
    // Lazy message (see checkBatchArguments): don't build "rows[i]" per row on the success path.
    for (int i = 0; i < count; i++) {
      if (rows[i] == null) {
        throw new NullPointerException("rows[" + i + "]");
      }
    }
  }

  private static void checkRowMajorBatchArguments(
      float[] query, float[] rowMajorMatrix, int rows, int cols, float[] out) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(rowMajorMatrix, "rowMajorMatrix");
    Objects.requireNonNull(out, "out");
    if (rows < 0) {
      throw new IllegalArgumentException("rows must be >= 0: " + rows);
    }
    if (cols < 0) {
      throw new IllegalArgumentException("cols must be >= 0: " + cols);
    }
    checkDimensions(query.length, cols);
    long required = (long) rows * cols;
    if (required > rowMajorMatrix.length) {
      throw new IllegalArgumentException(
          "rowMajorMatrix.length must be >= rows * cols: "
              + rowMajorMatrix.length
              + " < "
              + required);
    }
    if (out.length < rows) {
      throw new IllegalArgumentException(
          "out.length must be >= rows: " + out.length + " < " + rows);
    }
  }

  private static void checkExactStridedBatchArguments(
      float[] query,
      int queryOffset,
      float[] matrix,
      int matrixOffset,
      int rowStride,
      int rows,
      int columns,
      float[] out,
      int outOffset) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(matrix, "matrix");
    Objects.requireNonNull(out, "out");
    if (queryOffset < 0 || matrixOffset < 0 || outOffset < 0) {
      throw new IllegalArgumentException("query, matrix, and output offsets must be >= 0");
    }
    if (rows < 0 || columns < 0) {
      throw new IllegalArgumentException("rows and columns must be >= 0");
    }
    if (rowStride < columns) {
      throw new IllegalArgumentException(
          "rowStride must be >= columns: " + rowStride + " < " + columns);
    }
    if (queryOffset > query.length - columns) {
      throw new IllegalArgumentException("query range exceeds query.length");
    }
    long matrixEnd =
        rows == 0 ? matrixOffset : matrixOffset + (long) (rows - 1) * rowStride + columns;
    if (matrixEnd > matrix.length) {
      throw new IllegalArgumentException("strided matrix range exceeds matrix.length");
    }
    if (outOffset > out.length - rows) {
      throw new IllegalArgumentException("output range exceeds out.length");
    }
    if (out == query || out == matrix) {
      throw new IllegalArgumentException("out must not alias query or matrix");
    }
  }

  private static void checkGgufF32BatchArguments(
      float[] query, MemorySegment weight, int rows, int cols, float[] out) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(weight, "weight");
    Objects.requireNonNull(out, "out");
    if (rows < 0) {
      throw new IllegalArgumentException("rows must be >= 0: " + rows);
    }
    if (cols < 0) {
      throw new IllegalArgumentException("cols must be >= 0: " + cols);
    }
    checkDimensions(query.length, cols);
    long requiredBytes = (long) rows * cols * Float.BYTES;
    if (requiredBytes > weight.byteSize()) {
      throw new IllegalArgumentException(
          "weight byteSize must be >= rows * cols * 4: "
              + weight.byteSize()
              + " < "
              + requiredBytes);
    }
    if (out.length < rows) {
      throw new IllegalArgumentException(
          "out.length must be >= rows: " + out.length + " < " + rows);
    }
  }

  private static void checkSubVectorArguments(
      float[] out, int outOffset, float[] vector, int vectorOffset, int length) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(vector, "vector");
    if (outOffset < 0) {
      throw new IllegalArgumentException("outOffset must be >= 0: " + outOffset);
    }
    if (vectorOffset < 0) {
      throw new IllegalArgumentException("vectorOffset must be >= 0: " + vectorOffset);
    }
    if (length < 0) {
      throw new IllegalArgumentException("length must be >= 0: " + length);
    }
    if ((long) outOffset + length > out.length) {
      throw new IllegalArgumentException(
          "outOffset + length must be <= out.length: "
              + outOffset
              + " + "
              + length
              + " > "
              + out.length);
    }
    if ((long) vectorOffset + length > vector.length) {
      throw new IllegalArgumentException(
          "vectorOffset + length must be <= vector.length: "
              + vectorOffset
              + " + "
              + length
              + " > "
              + vector.length);
    }
  }

  private static void checkWeightedRowsArguments(
      float[] out,
      int outOffset,
      float[] matrix,
      int matrixOffset,
      int rowStride,
      float[] weights,
      int weightsOffset,
      int rows,
      int columns) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(matrix, "matrix");
    Objects.requireNonNull(weights, "weights");
    if (out == matrix || out == weights) {
      throw new IllegalArgumentException("matrix and weights must not alias out");
    }
    if (outOffset < 0 || matrixOffset < 0 || weightsOffset < 0) {
      throw new IllegalArgumentException("offsets must be >= 0");
    }
    if (rows < 0 || columns < 0) {
      throw new IllegalArgumentException("rows and columns must be >= 0");
    }
    if (rowStride < columns) {
      throw new IllegalArgumentException(
          "rowStride must be >= columns: " + rowStride + " < " + columns);
    }
    if ((long) outOffset + columns > out.length) {
      throw new IllegalArgumentException("output range exceeds out.length");
    }
    if ((long) weightsOffset + rows > weights.length) {
      throw new IllegalArgumentException("weight range exceeds weights.length");
    }
    long matrixEnd =
        rows == 0 ? matrixOffset : (long) matrixOffset + (long) (rows - 1) * rowStride + columns;
    if (matrixEnd > matrix.length) {
      throw new IllegalArgumentException("matrix range exceeds matrix.length");
    }
  }

  private static void checkGgufQuantizedDotArguments(
      float[] query, MemorySegment qWeight, long byteOffset, int dimensions, int blockBytes) {
    checkGgufQuantizedDotArguments(
        query, qWeight, byteOffset, dimensions, VectorUtilSupport.GGUF_Q_BLOCK_SIZE, blockBytes);
  }

  private static void checkGgufQuantizedDotArguments(
      float[] query,
      MemorySegment qWeight,
      long byteOffset,
      int dimensions,
      int blockSize,
      int blockBytes) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(qWeight, "qWeight");
    if (byteOffset < 0) {
      throw new IllegalArgumentException("byteOffset must be >= 0: " + byteOffset);
    }
    checkGgufQuantizedDimensions(query, dimensions, blockSize);
    long required = byteOffset + ggufQuantizedRowBytes(dimensions, blockSize, blockBytes);
    if (required < byteOffset || required > qWeight.byteSize()) {
      throw new IllegalArgumentException(
          "qWeight byteSize is too small for requested row: "
              + qWeight.byteSize()
              + " < "
              + required);
    }
  }

  private static void checkGgufQuantizedBatchArguments(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out, int blockBytes) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q_BLOCK_SIZE, blockBytes);
  }

  private static void checkGgufQuantizedBatchArguments(
      float[] query,
      MemorySegment qWeight,
      int rows,
      int cols,
      float[] out,
      int blockSize,
      int blockBytes) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(out, "out");
    checkDimensions(query.length, cols);
    checkGgufQuantizedMatrixArguments(qWeight, rows, cols, blockSize, blockBytes);
    if (out.length < rows) {
      throw new IllegalArgumentException(
          "out.length must be >= rows: " + out.length + " < " + rows);
    }
  }

  private static void checkGgufQuantizedBatchedArguments(
      float[] queries,
      MemorySegment qWeight,
      int batchSize,
      int rows,
      int cols,
      float[] out,
      int blockSize,
      int blockBytes) {
    Objects.requireNonNull(queries, "queries");
    Objects.requireNonNull(out, "out");
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1: " + batchSize);
    }
    checkGgufQuantizedMatrixArguments(qWeight, rows, cols, blockSize, blockBytes);
    int queryEntries = checkedProduct(batchSize, cols, "batchSize * cols");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    if (queries.length < queryEntries) {
      throw new IllegalArgumentException(
          "queries.length must be >= batchSize * cols: " + queries.length + " < " + queryEntries);
    }
    if (out.length < outputEntries) {
      throw new IllegalArgumentException(
          "out.length must be >= batchSize * rows: " + out.length + " < " + outputEntries);
    }
  }

  private static void checkGgufQuantizedMatrixArguments(
      MemorySegment qWeight, int rows, int cols, int blockSize, int blockBytes) {
    Objects.requireNonNull(qWeight, "qWeight");
    if (rows < 0) {
      throw new IllegalArgumentException("rows must be >= 0: " + rows);
    }
    if (cols < 0) {
      throw new IllegalArgumentException("cols must be >= 0: " + cols);
    }
    if (cols % blockSize != 0) {
      throw new IllegalArgumentException(
          "GGUF quantized dimensions must be a multiple of " + blockSize + ": " + cols);
    }
    long rowBytes = ggufQuantizedRowBytes(cols, blockSize, blockBytes);
    long required = rowBytes * rows;
    if (rows != 0 && required / rows != rowBytes) {
      throw new IllegalArgumentException("rows * rowBytes overflows: " + rows + " * " + rowBytes);
    }
    if (required > qWeight.byteSize()) {
      throw new IllegalArgumentException(
          "qWeight byteSize must be >= rows * rowBytes: " + qWeight.byteSize() + " < " + required);
    }
  }

  private static void checkGgufQuantizedDimensions(float[] query, int dimensions, int blockSize) {
    checkDimensions(query.length, dimensions);
    if (dimensions % blockSize != 0) {
      throw new IllegalArgumentException(
          "GGUF quantized dimensions must be a multiple of " + blockSize + ": " + dimensions);
    }
  }

  private static void checkGgufActivationScratch(
      byte[] q8Quants, float[] q8Scales, int dimensions, int blockSize) {
    Objects.requireNonNull(q8Quants, "q8Quants");
    Objects.requireNonNull(q8Scales, "q8Scales");
    if (q8Quants.length < dimensions) {
      throw new IllegalArgumentException(
          "q8Quants.length must be >= dimensions: " + q8Quants.length + " < " + dimensions);
    }
    int blocks = dimensions / blockSize;
    if (q8Scales.length < blocks) {
      throw new IllegalArgumentException(
          "q8Scales.length must be >= dimensions / "
              + blockSize
              + ": "
              + q8Scales.length
              + " < "
              + blocks);
    }
  }

  private static void checkGgufQ8KSums(short[] q8Sums, int dimensions) {
    Objects.requireNonNull(q8Sums, "q8Sums");
    int requiredSums = dimensions / VectorUtilSupport.GGUF_Q8_K_SUM_BLOCK_SIZE;
    if (q8Sums.length < requiredSums) {
      throw new IllegalArgumentException(
          "q8Sums.length must be >= dimensions / "
              + VectorUtilSupport.GGUF_Q8_K_SUM_BLOCK_SIZE
              + ": "
              + q8Sums.length
              + " < "
              + requiredSums);
    }
  }

  private static void checkGgufQ4CorrectionScratch(
      int[] q8ZeroPointCorrections, int activationEntries) {
    Objects.requireNonNull(q8ZeroPointCorrections, "q8ZeroPointCorrections");
    int required = activationEntries / 4;
    if (q8ZeroPointCorrections.length < required) {
      throw new IllegalArgumentException(
          "q8ZeroPointCorrections.length must be >= activation entries / 4: "
              + q8ZeroPointCorrections.length
              + " < "
              + required);
    }
  }

  private static void checkGgufQ4LaneScratch(
      float[] laneScratch, int batchSize, int rows, String label) {
    Objects.requireNonNull(laneScratch, "laneScratch");
    int outputEntries = checkedProduct(batchSize, rows, "batchSize * rows");
    int required = checkedProduct(outputEntries, 8, label);
    if (laneScratch.length < required) {
      throw new IllegalArgumentException(
          "lane scratch length must be >= batchSize * rows * 8: "
              + laneScratch.length
              + " < "
              + required);
    }
  }

  private static long ggufQuantizedRowBytes(int dimensions, int blockSize, int blockBytes) {
    return (long) (dimensions / blockSize) * blockBytes;
  }

  private static int checkedProduct(int left, int right, String label) {
    long product = (long) left * right;
    if (product > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(label + " exceeds maximum Java array length: " + product);
    }
    return (int) product;
  }
}
