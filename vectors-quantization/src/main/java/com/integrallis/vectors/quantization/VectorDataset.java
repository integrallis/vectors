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

import com.integrallis.vectors.core.VectorUtil;

/**
 * Read-only access to a collection of float vectors. Used as input for quantizer training and
 * encoding.
 *
 * <p>Implementations may be backed by in-memory arrays, a {@link
 * com.integrallis.vectors.storage.store.VectorStore}, or any other vector source.
 */
public interface VectorDataset {

  /** Returns the number of vectors in this dataset. */
  int size();

  /** Returns the dimensionality of each vector. */
  int dimension();

  /**
   * Returns the vector at the given ordinal. The returned array may be shared across calls; callers
   * must copy if they need to retain the data.
   *
   * @param ordinal the 0-based vector index
   * @return the float vector at the given position
   * @throws IndexOutOfBoundsException if ordinal is out of range
   */
  float[] getVector(int ordinal);

  /**
   * Computes the global centroid (mean vector) of this dataset. Provided as a shared default to
   * avoid duplicating the same loop across {@link BinaryQuantizer}, {@link RaBitQuantizer}, and
   * {@link ProductQuantizer}.
   *
   * @return a newly allocated float array of length {@link #dimension()} containing the mean vector
   */
  default float[] computeCentroid() {
    int dim = dimension();
    int n = size();
    float[] centroid = new float[dim];
    for (int i = 0; i < n; i++) {
      VectorUtil.addInPlace(centroid, getVector(i));
    }
    VectorUtil.scale(centroid, 1.0f / n);
    return centroid;
  }
}
