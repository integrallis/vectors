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
}
