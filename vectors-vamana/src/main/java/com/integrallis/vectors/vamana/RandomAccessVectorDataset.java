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
package com.integrallis.vectors.vamana;

import com.integrallis.vectors.quantization.VectorDataset;
import java.util.Objects;

/**
 * Adapter from {@link RandomAccessVectors} to {@link VectorDataset}. Allows training quantizers
 * directly from Vamana vector stores without creating an intermediate {@code float[][]}.
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
