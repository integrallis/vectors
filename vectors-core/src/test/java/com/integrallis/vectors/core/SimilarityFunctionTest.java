package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

/** Tests for {@link SimilarityFunction}. */
class SimilarityFunctionTest {

  @Test
  void euclidean_identicalVectors_scoreIsOne() {
    float[] v = {1.0f, 2.0f, 3.0f};
    assertThat(SimilarityFunction.EUCLIDEAN.compare(v, v)).isCloseTo(1.0f, within(1e-4f));
  }

  @Test
  void euclidean_resultInZeroOneRange() {
    float[] a = {1.0f, 0.0f, 0.0f};
    float[] b = {0.0f, 1.0f, 0.0f};
    float score = SimilarityFunction.EUCLIDEAN.compare(a, b);
    assertThat(score).isBetween(0.0f, 1.0f);
  }

  @Test
  void euclidean_farVectors_lowScore() {
    float[] a = {0.0f, 0.0f};
    float[] b = {100.0f, 100.0f};
    float score = SimilarityFunction.EUCLIDEAN.compare(a, b);
    assertThat(score).isLessThan(0.01f);
  }

  @Test
  void dotProduct_normalizedVectors_resultInZeroOneRange() {
    float[] a = {1.0f, 0.0f, 0.0f};
    float[] b = {0.0f, 1.0f, 0.0f};
    float score = SimilarityFunction.DOT_PRODUCT.compare(a, b);
    // Dot product of orthogonal unit vectors = 0, score = (1+0)/2 = 0.5
    assertThat(score).isCloseTo(0.5f, within(1e-4f));
  }

  @Test
  void dotProduct_sameDirection_highScore() {
    float[] a = {1.0f, 0.0f};
    float score = SimilarityFunction.DOT_PRODUCT.compare(a, a);
    // Dot of unit vector with itself = 1, score = (1+1)/2 = 1.0
    assertThat(score).isCloseTo(1.0f, within(1e-4f));
  }

  @Test
  void cosine_identicalDirection_highScore() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {2.0f, 4.0f, 6.0f};
    float score = SimilarityFunction.COSINE.compare(a, b);
    // Cosine of parallel vectors = 1, score = (1+1)/2 = 1.0
    assertThat(score).isCloseTo(1.0f, within(1e-4f));
  }

  @Test
  void cosine_resultInZeroOneRange() {
    float[] a = {1.0f, 0.0f};
    float[] b = {0.0f, 1.0f};
    float score = SimilarityFunction.COSINE.compare(a, b);
    // Cosine of orthogonal = 0, score = (1+0)/2 = 0.5
    assertThat(score).isCloseTo(0.5f, within(1e-4f));
  }

  @Test
  void maxInnerProduct_positiveScore() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 5.0f, 6.0f};
    float score = SimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(a, b);
    // dot = 32, score = 32 + 1 = 33
    assertThat(score).isCloseTo(33.0f, within(1e-3f));
  }

  @Test
  void maxInnerProduct_negativeScore() {
    float[] a = {1.0f, 0.0f};
    float[] b = {-5.0f, 0.0f};
    float score = SimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(a, b);
    // dot = -5, score = 1/(1-(-5)) = 1/6 ~= 0.1667
    assertThat(score).isCloseTo(1.0f / 6.0f, within(1e-4f));
  }

  @Test
  void scaleMaxInnerProductScore_monotonicOrdering() {
    // Verify that larger dot products always produce larger scores
    float[] scores = {-10f, -5f, -1f, -0.5f, 0f, 0.5f, 1f, 5f, 10f};
    for (int i = 0; i < scores.length - 1; i++) {
      float lower = SimilarityFunction.scaleMaxInnerProductScore(scores[i]);
      float upper = SimilarityFunction.scaleMaxInnerProductScore(scores[i + 1]);
      assertThat(upper).as("score(%f) > score(%f)", scores[i + 1], scores[i]).isGreaterThan(lower);
    }
  }

  // --- Byte vector similarity tests ---

  @Test
  void euclidean_bytes_identicalIsOne() {
    byte[] v = {1, 2, 3, 4};
    assertThat(SimilarityFunction.EUCLIDEAN.compare(v, v)).isCloseTo(1.0f, within(1e-4f));
  }

  @Test
  void cosine_bytes_resultInZeroOneRange() {
    byte[] a = {1, 2, 3};
    byte[] b = {4, 5, 6};
    float score = SimilarityFunction.COSINE.compare(a, b);
    assertThat(score).isBetween(0.0f, 1.0f);
  }
}
