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
 * Computes the similarity score between a query and a stored vector identified by node ID.
 *
 * <p>This abstraction decouples beam search scoring from the vector access mechanism, enabling both
 * full-precision and quantized scoring to share the same search algorithm.
 *
 * <p>Instances are not thread-safe and are created fresh per search call via {@link
 * NodeScorerFactory}.
 */
@FunctionalInterface
public interface NodeScorer {

  /**
   * Returns the similarity score for the stored vector at the given node ID.
   *
   * @param nodeId the 0-based node index in the graph
   * @return the similarity score (higher means more similar)
   */
  float score(int nodeId);

  /**
   * Scores up to {@code count} node IDs from {@code nodeIds[offset .. offset+count)} and writes the
   * results into {@code out[0 .. count)}.
   *
   * <p>The default implementation is a scalar loop. Scorer implementations that can take advantage
   * of fused batch SIMD (e.g. by aliasing references into a contiguous {@code float[][]} pool)
   * should override this with a bulk-friendly path.
   */
  default void bulkScore(int[] nodeIds, int offset, int count, float[] out) {
    for (int i = 0; i < count; i++) out[i] = score(nodeIds[offset + i]);
  }

  /**
   * Returns true if this scorer can provide neighbor-scoped batched scoring via {@link
   * #scoreNeighborBatch}. When true, the searcher may elect to score an origin's entire neighbor
   * list in a single call, typically to exploit a fused/packed per-node code layout (Fused ADC).
   */
  default boolean supportsNeighborBatch() {
    return false;
  }

  /**
   * Scores all {@code count} neighbors of {@code originId} at the layer this scorer was configured
   * for, writing the score for the i-th neighbor into {@code out[i]}. The ordering of the scores
   * must match the ordering of the origin's neighbor list as returned by {@code
   * HnswGraph#getNeighbors(originId, layer)}.
   *
   * <p>Only defined when {@link #supportsNeighborBatch()} returns true. Default throws.
   *
   * @param originId the node whose neighbor list is being scored
   * @param count size of the origin's neighbor list at the relevant layer
   * @param out output scores buffer; {@code out.length >= count} must hold
   */
  default void scoreNeighborBatch(int originId, int count, float[] out) {
    throw new UnsupportedOperationException("scoreNeighborBatch not supported by this NodeScorer");
  }
}
