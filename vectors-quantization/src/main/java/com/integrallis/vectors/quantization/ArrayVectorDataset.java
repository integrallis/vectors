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
