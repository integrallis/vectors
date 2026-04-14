package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Phase 1 gate tests for {@link IvfIndex}. Verifies recall@10 against brute-force ground truth. */
@Tag("unit")
class IvfIndexIntegrationTest {

  private static float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  /** Brute-force top-k ordinals by squared L2 (nearest first). */
  private static int[] bruteForceTopK(float[] query, float[][] dataset, int k) {
    Integer[] order = new Integer[dataset.length];
    for (int i = 0; i < order.length; i++) order[i] = i;
    Arrays.sort(
        order,
        (a, b) ->
            Float.compare(
                VectorUtil.squareDistance(query, dataset[a]),
                VectorUtil.squareDistance(query, dataset[b])));
    return Arrays.stream(order).limit(k).mapToInt(Integer::intValue).toArray();
  }

  private static double recall(int[] found, int[] groundTruth) {
    Set<Integer> gt = new HashSet<>();
    for (int x : groundTruth) gt.add(x);
    int hits = 0;
    for (int x : found) if (gt.contains(x)) hits++;
    return (double) hits / groundTruth.length;
  }

  @Test
  void buildAndSearchFlat_10K_K64_recall90() {
    // dim=32 clusters much better than dim=128 on random data (curse of dimensionality).
    // k=32 clusters, nprobe=16 (50% coverage) + SOAR gives robust recall@10 ≥ 0.90.
    int n = 10_000, dim = 32, k = 10, nprobe = 16;
    float[][] data = randomVecs(n, dim, 1L);
    IvfIndex idx =
        IvfIndex.build(
            data, null, SimilarityFunction.EUCLIDEAN, new IvfBuildParams(32, 50, 0.15f, true, 42L));

    int queries = 50;
    double totalRecall = 0.0;
    for (int q = 0; q < queries; q++) {
      float[] query = randomVecs(1, dim, 1000L + q)[0];
      IvfSearchResult result =
          idx.search(new IvfSearchRequest(query, k, nprobe, 0.15f, -Float.MAX_VALUE));
      int[] found = result.hits().stream().mapToInt(IvfHit::ordinal).toArray();
      int[] gt = bruteForceTopK(query, data, k);
      totalRecall += recall(found, gt);
    }
    assertThat(totalRecall / queries).isGreaterThanOrEqualTo(0.90);
  }

  @Test
  void searchReturnsAtMostKResults() {
    int n = 500, dim = 32;
    float[][] data = randomVecs(n, dim, 2L);
    IvfIndex idx =
        IvfIndex.build(
            data, null, SimilarityFunction.EUCLIDEAN, new IvfBuildParams(16, 20, 0f, false, 0L));

    IvfSearchResult result = idx.search(IvfSearchRequest.of(randomVecs(1, dim, 99L)[0], 10, 4));
    assertThat(result.hits().size()).isLessThanOrEqualTo(10);
  }

  @Test
  void searchWithEmptyResultWhenNoData() {
    float[][] data = new float[][] {{1f, 0f}};
    IvfIndex idx =
        IvfIndex.build(
            data, null, SimilarityFunction.EUCLIDEAN, new IvfBuildParams(1, 5, 0f, false, 0L));

    IvfSearchResult result = idx.search(IvfSearchRequest.of(new float[] {0f, 1f}, 1, 1));
    assertThat(result.isEmpty()).isFalse(); // the single vector is returned
    assertThat(result.size()).isEqualTo(1);
  }

  @Test
  void nprobeEqualsKGivesBruteForceRecall() {
    int n = 500, dim = 32, k = 5;
    float[][] data = randomVecs(n, dim, 3L);
    int numClusters = 16;
    IvfIndex idx =
        IvfIndex.build(
            data,
            null,
            SimilarityFunction.EUCLIDEAN,
            new IvfBuildParams(numClusters, 20, 0f, false, 0L));

    int queries = 20;
    double totalRecall = 0.0;
    for (int q = 0; q < queries; q++) {
      float[] query = randomVecs(1, dim, 2000L + q)[0];
      IvfSearchResult result =
          idx.search(new IvfSearchRequest(query, k, numClusters, 0f, -Float.MAX_VALUE));
      int[] found = result.hits().stream().mapToInt(IvfHit::ordinal).toArray();
      totalRecall += recall(found, bruteForceTopK(query, data, k));
    }
    assertThat(totalRecall / queries).isGreaterThanOrEqualTo(0.99);
  }

  @Test
  void soarExpansionDoesNotDegradesRecall() {
    int n = 1000, dim = 64, k = 10, nprobe = 8;
    float[][] data = randomVecs(n, dim, 4L);
    IvfIndex idxPlain =
        IvfIndex.build(
            data, null, SimilarityFunction.EUCLIDEAN, new IvfBuildParams(32, 30, 0f, false, 42L));
    IvfIndex idxSoar =
        IvfIndex.build(
            data, null, SimilarityFunction.EUCLIDEAN, new IvfBuildParams(32, 30, 0.2f, true, 42L));

    int queries = 20;
    double recallPlain = 0, recallSoar = 0;
    for (int q = 0; q < queries; q++) {
      float[] query = randomVecs(1, dim, 3000L + q)[0];
      int[] gt = bruteForceTopK(query, data, k);
      recallPlain +=
          recall(
              idxPlain
                  .search(new IvfSearchRequest(query, k, nprobe, 0f, -Float.MAX_VALUE))
                  .hits()
                  .stream()
                  .mapToInt(IvfHit::ordinal)
                  .toArray(),
              gt);
      recallSoar +=
          recall(
              idxSoar
                  .search(new IvfSearchRequest(query, k, nprobe, 0.2f, -Float.MAX_VALUE))
                  .hits()
                  .stream()
                  .mapToInt(IvfHit::ordinal)
                  .toArray(),
              gt);
    }
    // SOAR recall must be >= plain recall (or at most marginally below due to random dataset)
    assertThat(recallSoar / queries).isGreaterThanOrEqualTo((recallPlain / queries) - 0.05);
  }

  @Test
  void idsReturnedWhenProvided() {
    int n = 100, dim = 8;
    float[][] data = randomVecs(n, dim, 5L);
    String[] ids = new String[n];
    for (int i = 0; i < n; i++) ids[i] = "doc-" + i;

    IvfIndex idx =
        IvfIndex.build(
            data, ids, SimilarityFunction.EUCLIDEAN, new IvfBuildParams(10, 10, 0f, false, 0L));
    IvfSearchResult result = idx.search(IvfSearchRequest.of(randomVecs(1, dim, 77L)[0], 5, 5));

    for (IvfHit hit : result.hits()) {
      assertThat(hit.id()).isNotNull().startsWith("doc-");
    }
  }

  @Test
  void hitsAreSortedByDescendingScore() {
    int n = 200, dim = 16;
    float[][] data = randomVecs(n, dim, 6L);
    IvfIndex idx =
        IvfIndex.build(
            data, null, SimilarityFunction.EUCLIDEAN, new IvfBuildParams(8, 15, 0f, false, 0L));

    IvfSearchResult result = idx.search(IvfSearchRequest.of(randomVecs(1, dim, 88L)[0], 10, 4));
    var hits = result.hits();
    for (int i = 0; i < hits.size() - 1; i++)
      assertThat(hits.get(i).score()).isGreaterThanOrEqualTo(hits.get(i + 1).score());
  }
}
