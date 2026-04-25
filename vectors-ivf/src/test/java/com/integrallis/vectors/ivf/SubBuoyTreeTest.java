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
package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** P12: SubBuoyTree — hierarchical cluster routing with adaptive splitting. */
@Tag("unit")
class SubBuoyTreeTest {

  private static final int DIM = 32;
  private static final SimilarityFunction METRIC = SimilarityFunction.EUCLIDEAN;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  // Brute-force top-k ordinals by descending negative-L2 (nearest)
  private int[] bruteForceTopK(float[] query, float[][] data, int k) {
    record Pair(int idx, float dist) {}
    Pair[] pairs = new Pair[data.length];
    for (int i = 0; i < data.length; i++) {
      float d = 0f;
      for (int j = 0; j < query.length; j++) {
        float diff = query[j] - data[i][j];
        d += diff * diff;
      }
      pairs[i] = new Pair(i, d);
    }
    Arrays.sort(pairs, (a, b) -> Float.compare(a.dist(), b.dist()));
    int[] result = new int[k];
    for (int i = 0; i < k; i++) result[i] = pairs[i].idx();
    return result;
  }

  private double recall(int[] found, int[] gt) {
    Set<Integer> gtSet = new HashSet<>();
    for (int g : gt) gtSet.add(g);
    int hits = 0;
    for (int f : found) if (gtSet.contains(f)) hits++;
    return (double) hits / gt.length;
  }

  // ─── structure ───────────────────────────────────────────────────────────

  @Test
  void buildProducesRootWithKClusters() {
    float[][] vecs = randomVecs(500, DIM, 1L);
    IvfBuildParams params = new IvfBuildParams(8, 30, 0f, false, 42L, 0);
    // minSplitSize = 1000 → no splits
    ClusterSplitter splitter = new ClusterSplitter(1000, 30, 42L);
    SubBuoyTree tree = SubBuoyTree.build(vecs, null, METRIC, params, splitter);
    assertThat(tree.rootClusterCount()).isEqualTo(8);
  }

  @Test
  void leafCountEqualsKWhenNoSplits() {
    float[][] vecs = randomVecs(400, DIM, 2L);
    IvfBuildParams params = new IvfBuildParams(8, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(1000, 30, 42L);
    SubBuoyTree tree = SubBuoyTree.build(vecs, null, METRIC, params, splitter);
    assertThat(tree.leafCount()).isEqualTo(8);
  }

  @Test
  void depthIsOneWhenNoSplits() {
    float[][] vecs = randomVecs(400, DIM, 3L);
    IvfBuildParams params = new IvfBuildParams(8, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(1000, 30, 42L);
    SubBuoyTree tree = SubBuoyTree.build(vecs, null, METRIC, params, splitter);
    assertThat(tree.depth()).isEqualTo(1);
  }

  @Test
  void buildSplitsLargeCluster() {
    // 1000 vecs, k=4 clusters → ~250 per cluster; minSplitSize=50 → all should split
    float[][] vecs = randomVecs(1000, DIM, 4L);
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(50, 30, 42L);
    SubBuoyTree tree = SubBuoyTree.build(vecs, null, METRIC, params, splitter);
    // At least some clusters were split → leafCount > rootClusterCount
    assertThat(tree.leafCount()).isGreaterThan(tree.rootClusterCount());
  }

  @Test
  void depthIsGreaterThanOneWhenSplitsOccur() {
    float[][] vecs = randomVecs(1000, DIM, 5L);
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(50, 30, 42L);
    SubBuoyTree tree = SubBuoyTree.build(vecs, null, METRIC, params, splitter);
    assertThat(tree.depth()).isGreaterThan(1);
  }

  // ─── search ───────────────────────────────────────────────────────────────

  @Test
  void searchReturnsAtMostKResults() {
    float[][] vecs = randomVecs(500, DIM, 6L);
    IvfBuildParams params = new IvfBuildParams(8, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(1000, 30, 42L);
    SubBuoyTree tree = SubBuoyTree.build(vecs, null, METRIC, params, splitter);

    float[] query = randomVecs(1, DIM, 99L)[0];
    IvfSearchResult result = tree.search(new IvfSearchRequest(query, 10, 4, 0f, -Float.MAX_VALUE));
    assertThat(result.hits().size()).isLessThanOrEqualTo(10);
  }

  @Test
  void searchRecallExceedsThreshold() {
    // k=4, nprobe=3 (75%) → high recall, no splits needed
    int n = 2000, k = 10, nprobe = 3;
    float[][] data = randomVecs(n, DIM, 7L);
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(1000, 30, 42L);
    SubBuoyTree tree = SubBuoyTree.build(data, null, METRIC, params, splitter);

    int queries = 30;
    double totalRecall = 0.0;
    for (int q = 0; q < queries; q++) {
      float[] query = randomVecs(1, DIM, 1000L + q)[0];
      IvfSearchResult result =
          tree.search(new IvfSearchRequest(query, k, nprobe, 0f, -Float.MAX_VALUE));
      int[] found = result.hits().stream().mapToInt(IvfHit::ordinal).toArray();
      int[] gt = bruteForceTopK(query, data, k);
      totalRecall += recall(found, gt);
    }
    assertThat(totalRecall / queries).isGreaterThanOrEqualTo(0.85);
  }

  @Test
  void searchWithSplitClustersPreservesRecall() {
    // Splits should not hurt recall: both children are scanned on probe
    int n = 2000, k = 10, nprobe = 3;
    float[][] data = randomVecs(n, DIM, 8L);
    IvfBuildParams params = new IvfBuildParams(4, 30, 0f, false, 42L, 0);
    ClusterSplitter splitter = new ClusterSplitter(50, 30, 42L); // forces splits

    SubBuoyTree splitTree = SubBuoyTree.build(data, null, METRIC, params, splitter);
    SubBuoyTree flatTree =
        SubBuoyTree.build(data, null, METRIC, params, new ClusterSplitter(10000, 30, 42L));

    int queries = 20;
    double splitRecall = 0.0, flatRecall = 0.0;
    for (int q = 0; q < queries; q++) {
      float[] query = randomVecs(1, DIM, 2000L + q)[0];
      IvfSearchRequest req = new IvfSearchRequest(query, k, nprobe, 0f, -Float.MAX_VALUE);
      int[] gt = bruteForceTopK(query, data, k);
      splitRecall +=
          recall(splitTree.search(req).hits().stream().mapToInt(IvfHit::ordinal).toArray(), gt);
      flatRecall +=
          recall(flatTree.search(req).hits().stream().mapToInt(IvfHit::ordinal).toArray(), gt);
    }
    // Split tree recall should be within 15% of flat tree recall
    assertThat(splitRecall / queries).isGreaterThanOrEqualTo((flatRecall / queries) * 0.85);
  }
}
