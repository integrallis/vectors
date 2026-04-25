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

import java.util.Random;

/**
 * Generates random level assignments for HNSW nodes using the exponential distribution described in
 * the original paper.
 *
 * <p>Level = floor(-ln(uniform(0,1)) * mL) where mL = 1/ln(M).
 *
 * <p>For M=16: ~93.75% at level 0, ~5.8% at level 1, ~0.36% at level 2.
 *
 * <p>Levels are capped at {@link #MAX_LEVEL} to prevent pathological outliers: without a cap, the
 * extreme case {@code r = Double.MIN_VALUE} yields {@code -log(MIN_VALUE) ≈ 744}, giving a single
 * node a level of ~268 for M=16 (744 * 1/ln(16)). Such an outlier forces {@code 268} empty upper
 * layers and causes {@code HnswGraph.initNode()} to allocate {@code 268} {@link
 * com.integrallis.vectors.hnsw.NeighborArray} objects for a single node.
 */
final class RandomLevelGenerator {

  /**
   * Hard cap on the maximum assignable level. Covers practical datasets of up to ~10^13 nodes at
   * M=16 (expected max layer ≈ log_{M}(N) ≈ 11 for N=10^9). 32 leaves comfortable headroom.
   */
  static final int MAX_LEVEL = 32;

  private final double levelMultiplier; // mL = 1 / ln(M)
  private final Random random;

  RandomLevelGenerator(int maxConnections, long seed) {
    this.levelMultiplier = 1.0 / Math.log(maxConnections);
    this.random = new Random(seed);
  }

  /** Returns a random level in {@code [0, MAX_LEVEL]} drawn from the exponential distribution. */
  int nextLevel() {
    double r = random.nextDouble();
    // Avoid log(0) = -infinity; Double.MIN_VALUE is the smallest positive double.
    if (r == 0.0) {
      r = Double.MIN_VALUE;
    }
    return Math.min((int) (-Math.log(r) * levelMultiplier), MAX_LEVEL);
  }
}
