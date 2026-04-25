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

/**
 * In-memory implementation of {@link RandomAccessVectors} backed by a {@code float[][]}. Returns
 * direct references (no copying).
 */
public final class InMemoryVectors implements RandomAccessVectors {

  private final float[][] vectors;

  public InMemoryVectors(float[][] vectors) {
    if (vectors == null || vectors.length == 0) {
      throw new IllegalArgumentException("vectors must be non-null and non-empty");
    }
    int dim = vectors[0].length;
    for (int i = 1; i < vectors.length; i++) {
      if (vectors[i].length != dim) {
        throw new IllegalArgumentException(
            "All vectors must have the same dimension. "
                + "Vector 0 has dimension "
                + dim
                + " but vector "
                + i
                + " has dimension "
                + vectors[i].length);
      }
    }
    this.vectors = vectors;
  }

  @Override
  public int size() {
    return vectors.length;
  }

  @Override
  public int dimension() {
    return vectors[0].length;
  }

  @Override
  public float[] getVector(int ordinal) {
    return vectors[ordinal];
  }

  @Override
  public boolean sharesReturnBuffer() {
    return false;
  }
}
