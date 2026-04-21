package com.integrallis.vectors.core;

import java.lang.foreign.MemorySegment;

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
    IMPL.matVecDot(query, matrix, out, matrix.length);
  }

  /**
   * Fused GEMV dot product over the first {@code numRows} rows of {@code matrix}. Useful when
   * {@code matrix} is a reusable scratch buffer sized for a maximum batch but the current call
   * processes only a prefix.
   */
  public static void batchDotProduct(float[] query, float[][] matrix, float[] out, int numRows) {
    IMPL.matVecDot(query, matrix, out, numRows);
  }

  /**
   * Fills {@code out[i]} with the squared L2 distance from {@code query} to {@code matrix[i]} for
   * all rows {@code i} in {@code [0, matrix.length)}. {@code out.length} must be &ge; {@code
   * matrix.length}.
   *
   * <p>Delegates to {@link VectorUtilSupport#matVecSquaredL2} with a fused 4-row SIMD kernel.
   */
  public static void batchSquaredL2(float[] query, float[][] matrix, float[] out) {
    IMPL.matVecSquaredL2(query, matrix, out, matrix.length);
  }

  /**
   * Fused GEMV squared-L2 over the first {@code numRows} rows of {@code matrix}. Useful when {@code
   * matrix} is a reusable scratch buffer sized for a maximum batch but the current call processes
   * only a prefix.
   */
  public static void batchSquaredL2(float[] query, float[][] matrix, float[] out, int numRows) {
    IMPL.matVecSquaredL2(query, matrix, out, numRows);
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
}
