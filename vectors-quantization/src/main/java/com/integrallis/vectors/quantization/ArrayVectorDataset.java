package com.integrallis.vectors.quantization;

/**
 * A {@link VectorDataset} backed by an in-memory {@code float[][]} array. Useful for testing and
 * for datasets that fit entirely in memory.
 */
public final class ArrayVectorDataset implements VectorDataset {

  private final float[][] vectors;
  private final int dimension;

  /**
   * Creates a dataset from the given vectors. All vectors must have the same length.
   *
   * @param vectors the vector data (not copied; caller must not modify)
   * @throws IllegalArgumentException if the array is empty
   */
  public ArrayVectorDataset(float[][] vectors) {
    if (vectors.length == 0) {
      throw new IllegalArgumentException("Dataset must not be empty");
    }
    this.vectors = vectors;
    this.dimension = vectors[0].length;
  }

  @Override
  public int size() {
    return vectors.length;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public float[] getVector(int ordinal) {
    return vectors[ordinal];
  }
}
