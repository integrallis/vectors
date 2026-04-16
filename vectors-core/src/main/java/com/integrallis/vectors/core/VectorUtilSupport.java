package com.integrallis.vectors.core;

import java.lang.foreign.MemorySegment;

/**
 * Interface for all SIMD-accelerable vector operations. Implementations are selected at runtime by
 * {@link VectorizationProvider}: Panama Vector API (preferred) or scalar fallback.
 *
 * <p>Modeled after Apache Lucene's {@code VectorUtilSupport} and JVector's {@code
 * VectorUtilSupport}, combining the best patterns from both.
 */
public interface VectorUtilSupport {

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
}
