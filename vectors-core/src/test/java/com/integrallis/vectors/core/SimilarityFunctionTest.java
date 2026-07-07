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
package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for {@link SimilarityFunction}. */
@Tag("unit")
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

  // --- MemorySegment (zero-copy) vs float[] parity ---
  //
  // Proves the new compare(MemorySegment, MemorySegment, int) overload produces the SAME score as
  // compare(float[], float[]) for every metric, across several dims and random vectors. This is the
  // correctness contract that lets HnswSearcher SIMD-score directly from an mmap slice with no copy
  // and NOT regress recall.

  @Test
  void segmentCompare_matchesFloatArray_allMetrics() {
    int[] dims = {8, 128, 768};
    Random rnd = new Random(1234567L);
    try (Arena arena = Arena.ofConfined()) {
      for (int dim : dims) {
        for (int trial = 0; trial < 8; trial++) {
          float[] a = randomVector(rnd, dim);
          float[] b = randomVector(rnd, dim);
          MemorySegment segA = toSegment(arena, a);
          MemorySegment segB = toSegment(arena, b);

          for (SimilarityFunction sim : SimilarityFunction.values()) {
            float expected = sim.compare(a, b);
            float actual = sim.compare(segA, segB, dim);
            // Tight relative tolerance: same kernels + same float32 accumulation, so any diff is
            // pure fp reassociation noise, not a semantic divergence.
            float tol = Math.max(1e-5f, Math.abs(expected) * 1e-5f);
            assertThat(actual)
                .as("%s dim=%d trial=%d segment-vs-float[] parity", sim, dim, trial)
                .isCloseTo(expected, within(tol));
          }
        }
      }
    }
  }

  @Test
  void segmentCompare_identicalVectors_matchesFloatArray() {
    int dim = 128;
    Random rnd = new Random(42L);
    try (Arena arena = Arena.ofConfined()) {
      float[] v = randomVector(rnd, dim);
      MemorySegment seg = toSegment(arena, v);
      for (SimilarityFunction sim : SimilarityFunction.values()) {
        float expected = sim.compare(v, v);
        float actual = sim.compare(seg, seg, dim);
        assertThat(actual)
            .as("%s identical-vector segment parity", sim)
            .isCloseTo(expected, within(1e-5f));
      }
    }
  }

  private static float[] randomVector(Random rnd, int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rnd.nextFloat() * 2f - 1f; // [-1, 1)
    }
    return v;
  }

  private static MemorySegment toSegment(Arena arena, float[] v) {
    MemorySegment seg = arena.allocate((long) v.length * Float.BYTES);
    MemorySegment.copy(v, 0, seg, ValueLayout.JAVA_FLOAT, 0L, v.length);
    return seg;
  }
}
