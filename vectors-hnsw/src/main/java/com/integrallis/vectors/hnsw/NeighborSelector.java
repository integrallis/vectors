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
package com.integrallis.vectors.hnsw;

import com.integrallis.vectors.core.SimilarityFunction;

/**
 * Diversity-based neighbor selection (Algorithm 4 from the HNSW paper). Selects up to {@code
 * maxConnections} neighbors from a sorted candidate list, preferring candidates that are not
 * "blocked" by already-selected neighbors.
 *
 * <p>A candidate {@code e} is blocked if there exists an already-selected neighbor {@code r} such
 * that {@code sim(e, r) >= sim(e, query)} — meaning {@code r} is at least as close to {@code e} as
 * the query is, so {@code e} doesn't add useful connectivity.
 *
 * <p>After diverse selection, remaining slots are filled with blocked candidates (keepPruned=true)
 * to maximize connectivity.
 */
final class NeighborSelector {

  private NeighborSelector() {}

  /**
   * Selects diverse neighbors from candidates sorted by score descending (best first).
   *
   * @param candidates scored candidate list (sorted descending by score-to-query)
   * @param maxConnections maximum number of neighbors to select
   * @param vectors vector data for computing inter-neighbor distances
   * @param similarityFunction similarity metric
   * @return a new NeighborArray with the selected neighbors (sorted descending)
   */
  static NeighborArray selectDiverse(
      NeighborArray candidates,
      int maxConnections,
      RandomAccessVectors vectors,
      SimilarityFunction similarityFunction) {

    var result = new NeighborArray(maxConnections);
    if (candidates.size() == 0) {
      return result;
    }

    // If candidates fit, return them directly
    if (candidates.size() <= maxConnections) {
      result.copyFrom(candidates);
      return result;
    }

    // Scratch buffer for shared-buffer safety when comparing two vectors
    float[] scratch = new float[vectors.dimension()];

    // Track which candidates are blocked (pruned)
    boolean[] blocked = new boolean[candidates.size()];

    // Process candidates best-first (index 0 = best score)
    for (int i = 0; i < candidates.size() && result.size() < maxConnections; i++) {
      int candidateId = candidates.node(i);
      float scoreToQuery = candidates.score(i);

      // Copy candidate vector to scratch buffer (shared-buffer safety)
      float[] candidateVec = vectors.getVector(candidateId);
      System.arraycopy(candidateVec, 0, scratch, 0, scratch.length);

      boolean isBlocked = false;
      // Check against already-selected neighbors
      for (int j = 0; j < result.size(); j++) {
        int selectedId = result.node(j);
        float[] selectedVec = vectors.getVector(selectedId);
        float scoreER = similarityFunction.compare(scratch, selectedVec);

        if (scoreER >= scoreToQuery) {
          isBlocked = true;
          blocked[i] = true;
          break;
        }
      }

      if (!isBlocked) {
        result.insert(candidateId, scoreToQuery);
      }
    }

    // keepPruned=true: fill remaining slots with blocked candidates
    for (int i = 0; i < candidates.size() && result.size() < maxConnections; i++) {
      if (blocked[i]) {
        result.insert(candidates.node(i), candidates.score(i));
      }
    }

    return result;
  }
}
