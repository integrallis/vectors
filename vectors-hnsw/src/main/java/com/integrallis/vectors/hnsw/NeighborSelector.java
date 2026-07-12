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
import java.lang.foreign.MemorySegment;

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

    // Scratch buffer only needed for shared-buffer stores (e.g. mmap-backed) whose getVector()
    // may overwrite the previous return. Stable-array stores (InMemoryVectors) can compare against
    // getVector(id) directly, eliding the per-candidate copy.
    // Zero-copy segment path for mmap-backed stores (MappedBuildVectors): score stored-vs-stored
    // vectors directly off their slices, allocating no float[] per candidate — the same GC-avoidance
    // that makes the mmap build viable. Otherwise use the heap path (with a scratch copy for
    // shared-buffer stores whose getVector() aliases the previous return).
    boolean useSegments = vectors.supportsSegments();
    boolean sharedBuffer = vectors.sharesReturnBuffer();
    int dim = vectors.dimension();
    float[] scratch = (!useSegments && sharedBuffer) ? new float[dim] : null;

    // Track which candidates are blocked (pruned)
    boolean[] blocked = new boolean[candidates.size()];

    // Process candidates best-first (index 0 = best score)
    for (int i = 0; i < candidates.size() && result.size() < maxConnections; i++) {
      int candidateId = candidates.node(i);
      float scoreToQuery = candidates.score(i);

      MemorySegment candidateSeg = useSegments ? vectors.vectorSegment(candidateId) : null;
      float[] candidateVec = null;
      if (!useSegments) {
        candidateVec = vectors.getVector(candidateId);
        if (sharedBuffer) {
          // Copy candidate vector before subsequent getVector() calls alias the shared buffer.
          System.arraycopy(candidateVec, 0, scratch, 0, dim);
          candidateVec = scratch;
        }
      }

      boolean isBlocked = false;
      // Check against already-selected neighbors
      for (int j = 0; j < result.size(); j++) {
        int selectedId = result.node(j);
        float scoreER =
            useSegments
                ? similarityFunction.compare(candidateSeg, vectors.vectorSegment(selectedId), dim)
                : similarityFunction.compare(candidateVec, vectors.getVector(selectedId));

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
