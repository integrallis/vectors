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
}
