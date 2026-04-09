package com.integrallis.vectors.hnsw;

/**
 * Read-only access to vectors by ordinal. Decoupled from the quantization module's {@code
 * VectorDataset} to keep the HNSW module dependency-light.
 *
 * <p>Implementations may return a shared buffer from {@link #getVector(int)}; callers must not
 * retain the returned array across calls.
 */
public interface RandomAccessVectors {

  /** Returns the number of vectors. */
  int size();

  /** Returns the number of dimensions per vector. */
  int dimension();

  /**
   * Returns the vector at the given ordinal. The returned array may be a shared buffer; callers
   * must not retain it across calls.
   *
   * @param ordinal the 0-based vector index
   * @return the vector data
   */
  float[] getVector(int ordinal);
}
