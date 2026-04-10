package com.integrallis.vectors.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.bench.dataset.DatasetRegistry;
import com.integrallis.vectors.bench.dataset.FvecsLoader;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.quantization.ArrayVectorDataset;
import com.integrallis.vectors.quantization.ScalarQuantizedVectors;
import com.integrallis.vectors.quantization.ScalarQuantizer;
import com.integrallis.vectors.vamana.SearchResult;
import com.integrallis.vectors.vamana.VamanaIndex;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Regression guards for Vamana recall on the SIFT Small dataset (10K base, 100 queries, 128 dims,
 * L2). Tests multiple beam widths to anchor the recall–QPS tradeoff curve. Skipped when SIFT Small
 * files are absent.
 *
 * <p>Dataset path resolution is delegated to {@link DatasetRegistry}: set the {@code
 * VECTORS_BENCH_DATA} environment variable to override the default research/data/ location.
 */
@Tag("slow")
@EnabledIf("com.integrallis.vectors.bench.dataset.DatasetRegistry#siftSmallAvailable")
class SiftSmallVamanaRecallTest {

  private static float[][] base;
  private static float[][] queries;
  private static int[][] groundTruth;
  private static VamanaIndex index;

  @BeforeAll
  static void loadAndBuild() {
    Path dir = DatasetRegistry.siftSmallDir();
    base = FvecsLoader.readFvecs(dir.resolve("siftsmall_base.fvecs"));
    queries = FvecsLoader.readFvecs(dir.resolve("siftsmall_query.fvecs"));
    groundTruth = FvecsLoader.readIvecs(dir.resolve("siftsmall_groundtruth.ivecs"));

    index =
        VamanaIndex.builder(base, SimilarityFunction.EUCLIDEAN)
            .maxDegree(64)
            .searchListSize(128)
            .alpha(1.2f)
            .seed(42L)
            .build();
  }

  // -------------------------------------------------------------------------
  // Full-precision: recall floors at increasing beam widths
  // -------------------------------------------------------------------------

  @Test
  void fullPrecision_beamWidth64_recall10_above90() {
    double recall = meanRecall(10, 64);
    assertThat(recall)
        .as("Vamana full-precision L=64 recall@10 on SIFT Small")
        .isGreaterThanOrEqualTo(0.90);
  }

  @Test
  void fullPrecision_beamWidth128_recall10_above95() {
    double recall = meanRecall(10, 128);
    assertThat(recall)
        .as("Vamana full-precision L=128 recall@10 on SIFT Small")
        .isGreaterThanOrEqualTo(0.95);
  }

  @Test
  void fullPrecision_beamWidth200_recall10_above97() {
    double recall = meanRecall(10, 200);
    assertThat(recall)
        .as("Vamana full-precision L=200 recall@10 on SIFT Small")
        .isGreaterThanOrEqualTo(0.97);
  }

  // -------------------------------------------------------------------------
  // Two-pass SQ8 quantized search recall
  // -------------------------------------------------------------------------

  @Test
  void twoPassSq8_beamWidth128_recall10_above93() {
    ArrayVectorDataset dataset = new ArrayVectorDataset(base);
    ScalarQuantizer sq = ScalarQuantizer.train(dataset);
    ScalarQuantizedVectors cv = sq.encodeAll(dataset);
    index.enableQuantization(cv);
    try {
      double recall = meanRecallTwoPass(10, 128, 2.0f);
      assertThat(recall)
          .as("Vamana SQ8 two-pass L=128 overQuery=2x recall@10 on SIFT Small")
          .isGreaterThanOrEqualTo(0.93);
    } finally {
      index.disableQuantization();
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private static double meanRecall(int k, int searchListSize) {
    int[][] approx = new int[queries.length][];
    for (int i = 0; i < queries.length; i++) {
      SearchResult r = index.search(queries[i], k, searchListSize);
      approx[i] = r.nodeIds();
    }
    return RecallUtil.meanRecallAtK(groundTruth, approx, k);
  }

  private static double meanRecallTwoPass(int k, int searchListSize, float overQueryFactor) {
    int[][] approx = new int[queries.length][];
    for (int i = 0; i < queries.length; i++) {
      SearchResult r = index.searchTwoPass(queries[i], k, searchListSize, overQueryFactor);
      approx[i] = r.nodeIds();
    }
    return RecallUtil.meanRecallAtK(groundTruth, approx, k);
  }
}
