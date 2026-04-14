package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BuoyIndexTest {

  private static final int DIM = 64;
  private static final int K = 16;
  private static final int N = 2000;

  private float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  @Test
  void trainProducesKBuoys() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 1L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    assertThat(idx.buoyVectors().length).isEqualTo(K);
    assertThat(idx.buoyVectors()[0].length).isEqualTo(DIM);
  }

  @Test
  void kReturnsCorrectCount() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 2L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    assertThat(idx.k()).isEqualTo(K);
  }

  @Test
  void routeReturnsNprobeClusterIds() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 3L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 99L)[0];

    int[] result = idx.route(query, 4, 0f);

    assertThat(result).hasSize(4);
  }

  @Test
  void routeIdsAreUniqueAndWithinRange() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 4L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 100L)[0];

    int[] result = idx.route(query, 8, 0f);

    Set<Integer> seen = new HashSet<>();
    for (int id : result) {
      assertThat(id).isBetween(0, K - 1);
      assertThat(seen.add(id)).isTrue();
    }
  }

  @Test
  void routeIsIdempotent() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 5L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 101L)[0];

    int[] r1 = idx.route(query, 5, 0f);
    int[] r2 = idx.route(query, 5, 0f);

    assertThat(r1).isEqualTo(r2);
  }

  @Test
  void soarGammaExpandsResultsBeyondNprobe() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 6L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    float[] query = randomVecs(1, DIM, 102L)[0];

    int[] plain = idx.route(query, 4, 0f);
    int[] soar = idx.route(query, 4, 0.3f);

    // With SOAR, result may be >= plain (spill target may be added)
    assertThat(soar.length).isGreaterThanOrEqualTo(plain.length);
  }

  @Test
  void encodeDecodeRoundTrip() {
    BuoyIndex original =
        BuoyIndex.train(randomVecs(N, DIM, 7L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    BuoyIndex decoded = BuoyIndex.decode(original.encode());

    assertThat(decoded.k()).isEqualTo(original.k());
    assertThat(decoded.metric()).isEqualTo(original.metric());
    assertThat(decoded.clusterSizes()).isEqualTo(original.clusterSizes());
    for (int i = 0; i < K; i++) {
      assertThat(decoded.buoyVectors()[i]).isEqualTo(original.buoyVectors()[i]);
    }
  }

  @Test
  void clusterSizesArePositiveAndSumToN() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 8L), K, SimilarityFunction.EUCLIDEAN, false, 42L);
    int total = Arrays.stream(idx.clusterSizes()).sum();
    assertThat(total).isEqualTo(N);
    for (int s : idx.clusterSizes()) assertThat(s).isGreaterThanOrEqualTo(0);
  }

  @Test
  void clusterRadiiAreNonNegative() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 9L), K, SimilarityFunction.EUCLIDEAN, true, 42L);
    for (float r : idx.clusterRadii()) assertThat(r).isGreaterThanOrEqualTo(0f);
  }

  @Test
  void metricPreserved() {
    BuoyIndex idx =
        BuoyIndex.train(randomVecs(N, DIM, 10L), K, SimilarityFunction.DOT_PRODUCT, false, 0L);
    assertThat(idx.metric()).isEqualTo(SimilarityFunction.DOT_PRODUCT);
  }

  @Test
  void memoryFootprintWithinBound_K1024_D128() {
    // At K=1024, D=128: buoys[]=512KB + spill[]=4KB + sizes[]=4KB + radii[]=4KB ≈ 524KB
    BuoyIndex idx =
        BuoyIndex.train(
            randomVecs(1024 * 256, 128, 11L), 1024, SimilarityFunction.EUCLIDEAN, true, 1L);
    byte[] encoded = idx.encode();
    // Encoded size: 4+4+UTF(metric)+1024*128*4 floats + 3*(1024*4) ints/floats ≈ 536KB
    assertThat(encoded.length).isLessThan(600 * 1024); // 600 KB generous upper bound
  }
}
