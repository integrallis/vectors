package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.quantization.VectorDataset;
import java.util.Objects;

/**
 * Adapter from {@link RandomAccessVectors} to {@link VectorDataset}. Allows training quantizers
 * directly from HNSW vector stores without creating an intermediate {@code float[][]}.
 */
public final class RandomAccessVectorDataset implements VectorDataset {

  private final RandomAccessVectors vectors;

  /**
   * Wraps a {@link RandomAccessVectors} instance as a {@link VectorDataset}.
   *
   * @param vectors the vectors to wrap; must not be null
   */
  public RandomAccessVectorDataset(RandomAccessVectors vectors) {
    this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
  }

  @Override
  public int size() {
    return vectors.size();
  }

  @Override
  public int dimension() {
    return vectors.dimension();
  }

  @Override
  public float[] getVector(int ordinal) {
    return vectors.getVector(ordinal);
  }
}
