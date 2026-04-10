package com.integrallis.vectors.vamana;

/**
 * Creates a {@link NodeScorer} for a given query vector. Each search call creates a fresh scorer,
 * ensuring per-thread isolation of any internal state (e.g., quantized lookup tables).
 *
 * <p>Two common implementations:
 *
 * <ul>
 *   <li><b>Full-precision</b>: {@code query -> nodeId -> sim.compare(query,
 *       vectors.getVector(nodeId))}
 *   <li><b>Quantized</b>: {@code query -> { ScoreFunction sf = compressed.scoreFunctionFor(query,
 *       sim); return sf::score; }}
 * </ul>
 */
@FunctionalInterface
public interface NodeScorerFactory {

  /**
   * Creates a scorer for the given query vector.
   *
   * @param query the query vector
   * @return a scorer that computes similarity against stored vectors
   */
  NodeScorer scorer(float[] query);
}
