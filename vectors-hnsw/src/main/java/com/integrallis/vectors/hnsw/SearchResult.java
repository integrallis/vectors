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

/**
 * Search result containing ranked node IDs and their similarity scores. Results are ordered by
 * score descending (best first).
 *
 * <p><b>Mutability note:</b> The {@code nodeIds} and {@code scores} arrays are <em>not</em>
 * defensively copied. Callers must not mutate them; doing so would corrupt any cached or shared
 * result state. To hold results beyond the lifetime of the search, copy the arrays explicitly.
 *
 * @param nodeIds the node IDs, ranked by score descending; must not be mutated by callers
 * @param scores the similarity scores, ranked descending; must not be mutated by callers
 */
public record SearchResult(int[] nodeIds, float[] scores) {

  /** Returns the number of results. */
  public int size() {
    return nodeIds.length;
  }

  /** Returns the node ID at the given rank (0 = best). */
  public int nodeId(int rank) {
    return nodeIds[rank];
  }

  /** Returns the score at the given rank (0 = best). */
  public float score(int rank) {
    return scores[rank];
  }
}
