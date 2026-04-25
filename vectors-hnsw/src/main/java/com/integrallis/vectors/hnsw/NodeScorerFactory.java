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
