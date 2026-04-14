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
   */
  public static void batchDotProduct(float[] query, float[][] matrix, float[] out) {
    for (int i = 0; i < matrix.length; i++) {
      out[i] = IMPL.dotProduct(query, 0, matrix[i], 0, query.length);
    }
  }

  /**
   * Fills {@code out[i]} with the squared L2 distance from {@code query} to {@code matrix[i]} for
   * all rows {@code i} in {@code [0, matrix.length)}. {@code out.length} must be &ge; {@code
   * matrix.length}.
   */
  public static void batchSquaredL2(float[] query, float[][] matrix, float[] out) {
    for (int i = 0; i < matrix.length; i++) {
      out[i] = IMPL.squareDistance(query, 0, matrix[i], 0, query.length);
    }
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
