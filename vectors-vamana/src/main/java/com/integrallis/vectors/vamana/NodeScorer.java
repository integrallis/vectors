package com.integrallis.vectors.vamana;

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
}
