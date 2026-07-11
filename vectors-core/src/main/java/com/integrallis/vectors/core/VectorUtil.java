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

  /** Batched row-major GEMV over GGUF Q4_0 rows. */
  public static void ggufQ4_0BatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q4_0_BLOCK_BYTES);
    IMPL.ggufQ4_0MatVecDot(query, qWeight, rows, cols, out);
  }

  /** Batched row-major GEMV over GGUF Q8_0 rows. */
  public static void ggufQ8_0BatchDotProduct(
      float[] query, MemorySegment qWeight, int rows, int cols, float[] out) {
    checkGgufQuantizedBatchArguments(
        query, qWeight, rows, cols, out, VectorUtilSupport.GGUF_Q8_0_BLOCK_BYTES);
    IMPL.ggufQ8_0MatVecDot(query, qWeight, rows, cols, out);
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
    Objects.requireNonNull(qWeight, "qWeight");
    Objects.requireNonNull(out, "out");
    if (rows < 0) {
      throw new IllegalArgumentException("rows must be >= 0: " + rows);
    }
    checkGgufQuantizedDimensions(query, cols, blockSize);
    if (out.length < rows) {
      throw new IllegalArgumentException(
          "out.length must be >= rows: " + out.length + " < " + rows);
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

  private static void checkGgufQuantizedDimensions(float[] query, int dimensions) {
    checkGgufQuantizedDimensions(query, dimensions, VectorUtilSupport.GGUF_Q_BLOCK_SIZE);
  }

  private static void checkGgufQuantizedDimensions(float[] query, int dimensions, int blockSize) {
    checkDimensions(query.length, dimensions);
    if (dimensions % blockSize != 0) {
      throw new IllegalArgumentException(
          "GGUF quantized dimensions must be a multiple of " + blockSize + ": " + dimensions);
    }
  }

  private static long ggufQuantizedRowBytes(int dimensions, int blockBytes) {
    return ggufQuantizedRowBytes(dimensions, VectorUtilSupport.GGUF_Q_BLOCK_SIZE, blockBytes);
  }

  private static long ggufQuantizedRowBytes(int dimensions, int blockSize, int blockBytes) {
    return (long) (dimensions / blockSize) * blockBytes;
  }
}
