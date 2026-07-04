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
import java.util.Objects;
import java.util.Random;

/**
 * Two-pass batch construction for the Vamana graph index (DiskANN, NeurIPS 2019).
 *
 * <p>Build flow:
 *
 * <ol>
 *   <li>Compute medoid: centroid → find nearest point (linear scan)
 *   <li>Initialize all nodes
 *   <li>Initialize random graph: R random neighbors per node with computed scores
 *   <li>Pass 1 (alpha=1.0): for each node (random order), beam search → robust prune → backlinks
 *   <li>Pass 2 (alpha): same with configured alpha on the improved graph
 * </ol>
 */
public final class VamanaGraphBuilder {

  private final int maxDegree;
  private final int searchListSize;
  private final float alpha;
  private final RandomAccessVectors vectors;
  private final SimilarityFunction sim;
  private final Random random;

  private VamanaGraphBuilder(
      int maxDegree,
      int searchListSize,
      float alpha,
      RandomAccessVectors vectors,
      SimilarityFunction sim,
      long seed) {
    this.maxDegree = maxDegree;
    this.searchListSize = searchListSize;
    this.alpha = alpha;
    this.vectors = Objects.requireNonNull(vectors, "vectors must not be null");
    this.sim = Objects.requireNonNull(sim, "sim must not be null");
    this.random = new Random(seed);
  }

  /**
   * Creates a new builder.
   *
   * @param maxDegree maximum degree R
   * @param searchListSize search list size L for beam search during construction
   * @param alpha diversity parameter for robust pruning
   * @param vectors the dataset
   * @param sim similarity function
   * @param seed random seed for deterministic construction
   */
  public static VamanaGraphBuilder create(
      int maxDegree,
      int searchListSize,
      float alpha,
      RandomAccessVectors vectors,
      SimilarityFunction sim,
      long seed) {
    if (maxDegree <= 0) {
      throw new IllegalArgumentException("maxDegree must be positive: " + maxDegree);
    }
    if (searchListSize <= 0) {
      throw new IllegalArgumentException("searchListSize must be positive: " + searchListSize);
    }
    if (alpha < 1.0f) {
      throw new IllegalArgumentException("alpha must be >= 1.0: " + alpha);
    }
    return new VamanaGraphBuilder(maxDegree, searchListSize, alpha, vectors, sim, seed);
  }

  /** Builds the Vamana graph and returns it. */
  public VamanaGraph build() {
    int n = vectors.size();
    var graph = new VamanaGraph(n, maxDegree);

    // Initialize all nodes
    for (int i = 0; i < n; i++) {
      graph.initNode(i);
    }

    // Compute and set medoid
    int medoid = computeMedoid();
    graph.setMedoid(medoid);

    // Initialize random graph
    initializeRandomGraph(graph);

    // Pass 1: alpha = 1.0 (strict diversity)
    buildPass(graph, 1.0f);

    // Pass 2: alpha = configured value (relaxed diversity for long-range edges)
    if (alpha > 1.0f) {
      buildPass(graph, alpha);
    }

    return graph;
  }

  /**
   * Computes the medoid — the dataset point closest to the centroid in Euclidean space.
   *
   * <p>The medoid is a <em>geometric</em> property of the dataset (the point that minimises the
   * total L2 distance to all other points). It is always computed using squared L2 distance,
   * regardless of the index's configured similarity function. Using the configured function (e.g.,
   * {@code DOT_PRODUCT} or {@code COSINE}) produces geometrically incorrect results for
   * non-normalised or non-centred data: the dot-product maximiser of the centroid is not the
   * L2-nearest point to the centroid.
   *
   * <p><b>Visibility:</b> package-private solely to allow direct unit testing from {@code
   * VamanaGraphBuilderTest}; not part of the public API and must not be promoted to {@code public}.
   *
   * @return the index of the medoid
   */
  // @VisibleForTesting
  int computeMedoid() {
    int n = vectors.size();
    int dim = vectors.dimension();

    // Compute centroid
    float[] centroid = new float[dim];
    for (int i = 0; i < n; i++) {
      float[] vec = vectors.getVector(i);
      for (int d = 0; d < dim; d++) {
        centroid[d] += vec[d];
      }
    }
    for (int d = 0; d < dim; d++) {
      centroid[d] /= n;
    }

    // Find nearest point to centroid using squared L2 distance — always, regardless of
    // the index's similarity function (see Javadoc above).
    int bestNode = 0;
    float bestDistSq = l2SquaredDistance(centroid, vectors.getVector(0), dim);
    for (int i = 1; i < n; i++) {
      float distSq = l2SquaredDistance(centroid, vectors.getVector(i), dim);
      if (distSq < bestDistSq) {
        bestDistSq = distSq;
        bestNode = i;
      }
    }
    return bestNode;
  }

  private static float l2SquaredDistance(float[] a, float[] b, int dim) {
    float sum = 0f;
    for (int d = 0; d < dim; d++) {
      float diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }

  /** Initializes each node with R random neighbors (no self-loops). */
  private void initializeRandomGraph(VamanaGraph graph) {
    int n = vectors.size();
    int numRandomNeighbors = Math.min(maxDegree, n - 1);

    // Pre-allocate the candidates array once outside the node loop.
    // Allocating new int[n-1] inside the loop produces O(n²) total allocation:
    // for n=10,000 that is ~400 MB of short-lived arrays; for n=100,000 it is ~40 GB.
    int[] candidates = n > 1 ? new int[n - 1] : new int[0];

    for (int node = 0; node < n; node++) {
      var neighbors = graph.getNeighbors(node);

      // Refill candidates with all non-self indices (resets any shuffle from prior iteration).
      int idx = 0;
      for (int i = 0; i < n; i++) {
        if (i != node) candidates[idx++] = i;
      }

      // Fisher-Yates partial shuffle — pick numRandomNeighbors unique neighbors.
      for (int i = 0; i < numRandomNeighbors; i++) {
        int j = i + random.nextInt(candidates.length - i);
        int tmp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = tmp;

        int neighborId = candidates[i];
        float score = sim.compare(vectors.getVector(node), vectors.getVector(neighborId));
        neighbors.insert(neighborId, score);
      }
    }
  }

  /** Performs one pass of Vamana construction with the given alpha. */
  private void buildPass(VamanaGraph graph, float passAlpha) {
    int n = vectors.size();

    // Random permutation of node IDs
    int[] order = new int[n];
    for (int i = 0; i < n; i++) {
      order[i] = i;
    }
    for (int i = n - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    // Pre-allocate scratch buffers for this pass
    var searcher = new VamanaSearcher(graph, vectors, sim);
    var pruneCandidates = new NeighborArray(searchListSize + maxDegree + 1);
    var pruneResult = new NeighborArray(maxDegree + 1);
    // maxDegree existing neighbors + 1 new backlink = maxDegree+1 entries at most.
    var backlinkCandidates = new NeighborArray(maxDegree + 1);
    var backlinkResult = new NeighborArray(maxDegree + 1);

    for (int iter = 0; iter < n; iter++) {
      int node = order[iter];
      float[] query = vectors.getVector(node);

      // Beam search from medoid to find candidates
      SearchResult searchResult = searcher.search(query, searchListSize, searchListSize);

      // Merge search results with existing neighbors into candidate set
      pruneCandidates.clear();
      // Add search results
      for (int i = 0; i < searchResult.size(); i++) {
        int candidateId = searchResult.nodeId(i);
        if (candidateId != node) {
          pruneCandidates.addUnsorted(candidateId, searchResult.score(i));
        }
      }
      // Add existing neighbors
      var existingNeighbors = graph.getNeighbors(node);
      for (int i = 0; i < existingNeighbors.size(); i++) {
        int neighborId = existingNeighbors.node(i);
        if (neighborId != node && !pruneCandidates.contains(neighborId)) {
          pruneCandidates.addUnsorted(
              neighborId, sim.compare(query, vectors.getVector(neighborId)));
        }
      }
      pruneCandidates.sort();

      // Robust prune
      RobustPruner.robustPrune(
          node, pruneCandidates, maxDegree, passAlpha, vectors, sim, pruneResult);

      // Replace node's neighbor list with pruned result
      existingNeighbors.clear();
      existingNeighbors.copyFrom(pruneResult);

      // Add backlinks: for each new neighbor p, add node to p's neighbor list
      for (int i = 0; i < pruneResult.size(); i++) {
        int neighbor = pruneResult.node(i);
        var neighborNeighbors = graph.getNeighbors(neighbor);

        if (!neighborNeighbors.contains(node)) {
          // Reuse the pruned score: the metric is symmetric, so compare(neighbor, node) is
          // bit-identical to the score already carried in pruneResult (compare(node, neighbor)).
          float backlinkScore = pruneResult.score(i);

          if (neighborNeighbors.size() < maxDegree) {
            // Room available: just insert
            neighborNeighbors.insert(node, backlinkScore);
          } else {
            // Neighbor list full: merge and prune
            backlinkCandidates.clear();
            for (int j = 0; j < neighborNeighbors.size(); j++) {
              backlinkCandidates.addUnsorted(neighborNeighbors.node(j), neighborNeighbors.score(j));
            }
            backlinkCandidates.addUnsorted(node, backlinkScore);
            backlinkCandidates.sort();

            RobustPruner.robustPrune(
                neighbor, backlinkCandidates, maxDegree, passAlpha, vectors, sim, backlinkResult);

            neighborNeighbors.clear();
            neighborNeighbors.copyFrom(backlinkResult);
          }
        }
      }
    }
  }
}
