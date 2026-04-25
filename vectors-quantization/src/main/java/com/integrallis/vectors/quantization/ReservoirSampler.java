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

import java.util.Random;

/**
 * Vitter's Algorithm R reservoir sampling for selecting a random subset of indices from a dataset.
 *
 * <p>This is the single authoritative implementation shared by {@link ScalarQuantizer} and {@link
 * ProductQuantizer}, which both need to sample training vectors from potentially large datasets.
 *
 * <p>Given {@code datasetSize} items and a reservoir of size {@code maxSamples}, returns an {@code
 * int[]} of length {@code min(datasetSize, maxSamples)} where each entry is a selected dataset
 * index. The selection is uniformly random over all {@code C(datasetSize, sampleSize)} subsets.
 */
final class ReservoirSampler {

  private ReservoirSampler() {}

  /**
   * Returns selected dataset indices using Vitter's Algorithm R.
   *
   * @param datasetSize total number of items in the dataset
   * @param maxSamples maximum reservoir size
   * @param rng the random number generator to use
   * @return array of length {@code min(datasetSize, maxSamples)} containing selected indices in
   *     reservoir order (not sorted)
   */
  static int[] sampleIndices(int datasetSize, int maxSamples, Random rng) {
    int sampleSize = Math.min(datasetSize, maxSamples);
    int[] indices = new int[sampleSize];

    // Fill the reservoir with the first sampleSize items
    for (int i = 0; i < sampleSize; i++) {
      indices[i] = i;
    }

    // For each remaining item, randomly replace a reservoir slot
    for (int i = sampleSize; i < datasetSize; i++) {
      int j = rng.nextInt(i + 1);
      if (j < sampleSize) {
        indices[j] = i;
      }
    }

    return indices;
  }
}
