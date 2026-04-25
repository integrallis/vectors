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
package com.integrallis.vectors.vamana;

import com.integrallis.vectors.core.SimilarityFunction;

/**
 * Alpha-parameterized diversity pruning from the DiskANN paper (Algorithm 2). Selects a subset of
 * candidates that provides both proximity and angular diversity.
 *
 * <p><b>Pruning condition</b> (in similarity space): candidate {@code c} is "covered" by already
 * selected neighbor {@code p*} if:
 *
 * <pre>   sim(c, p*) &gt; sim(baseNode, c) * alpha</pre>
 *
 * <p><b>Single-pass with ratio check:</b> Candidates are scanned once in score order (best first).
 * Each candidate's inter-neighbor similarity is computed once and compared against the alpha
 * threshold. This is the canonical implementation of Algorithm 2 from the DiskANN paper.
 *
 * <ul>
 *   <li>At alpha=1.0: equivalent to HNSW's diversity heuristic (strict diversity)
 *   <li>At alpha&gt;1.0: harder to trigger pruning → more diverse/long-range edges survive
 * </ul>
 */
final class RobustPruner {

  private RobustPruner() {}

  /**
   * Performs robust pruning on candidate neighbors for a base node.
   *
   * @param baseNode the node being pruned
   * @param candidates candidate neighbors sorted descending by score to baseNode; may include
   *     baseNode itself (will be skipped)
   * @param maxDegree maximum number of neighbors to retain (R)
   * @param targetAlpha diversity parameter; 1.0 = strict, &gt;1.0 = more diverse
   * @param vectors vector data for computing inter-candidate distances
   * @param sim similarity function
   * @param result pre-allocated NeighborArray to store results (will be cleared first)
   */
  static void robustPrune(
      int baseNode,
      NeighborArray candidates,
      int maxDegree,
      float targetAlpha,
      RandomAccessVectors vectors,
      SimilarityFunction sim,
      NeighborArray result) {

    result.clear();

    if (candidates.size() == 0) {
      return;
    }

    int n = candidates.size();
    // Track which candidates are still eligible (not pruned and not baseNode)
    boolean[] eligible = new boolean[n];
    int eligibleCount = 0;
    for (int i = 0; i < n; i++) {
      if (candidates.node(i) != baseNode) {
        eligible[i] = true;
        eligibleCount++;
      }
    }

    if (eligibleCount == 0) {
      return;
    }

    // Scratch buffer: copy candidateVec before calling getVector(selectedNode) to guard
    // against shared-buffer RandomAccessVectors implementations (e.g., VectorStoreVectors)
    // that overwrite the same float[] on each call — without this copy, both calls alias
    // the same array and sim.compare(x, x) prunes every candidate after the first.
    float[] candidateScratch = new float[vectors.dimension()];

    // Single-pass: scan candidates in score order (best first), check coverage at targetAlpha
    for (int i = 0; i < n; i++) {
      if (!eligible[i]) {
        continue;
      }
      if (result.size() >= maxDegree) {
        break;
      }

      int candidateNode = candidates.node(i);
      float candidateScore = candidates.score(i);

      // Copy candidate vector to scratch before calling getVector(selectedNode):
      // shared-buffer implementations overwrite the returned array on each call.
      float[] candidateVec = vectors.getVector(candidateNode);
      System.arraycopy(candidateVec, 0, candidateScratch, 0, candidateScratch.length);

      // Check if this candidate is covered by any already-selected neighbor
      boolean covered = false;
      for (int j = 0; j < result.size(); j++) {
        int selectedNode = result.node(j);
        float simCandidateSelected = sim.compare(candidateScratch, vectors.getVector(selectedNode));
        if (simCandidateSelected > candidateScore * targetAlpha) {
          covered = true;
          break;
        }
      }

      if (!covered) {
        result.insert(candidateNode, candidateScore);
        eligible[i] = false;
      }
    }

    // Fill remaining slots with best unselected candidates (keepPruned=true)
    for (int i = 0; i < n && result.size() < maxDegree; i++) {
      if (eligible[i]) {
        result.insert(candidates.node(i), candidates.score(i));
      }
    }
  }
}
