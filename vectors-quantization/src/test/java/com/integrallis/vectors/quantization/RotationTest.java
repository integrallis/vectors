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
package com.integrallis.vectors.quantization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.vectors.core.VectorUtil;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Rotation} implementations: {@link GivensRotation}, {@link QuaternionRotation},
 * and the {@link Rotation} interface contract.
 */
class RotationTest {

  // --- Shared helpers ---

  private static float[] randomVector(int dim, Random rng) {
    float[] v = new float[dim];
    for (int d = 0; d < dim; d++) {
      v[d] = (float) rng.nextGaussian();
    }
    return v;
  }

  private static float norm(float[] v) {
    return (float) Math.sqrt(VectorUtil.dotProduct(v, v));
  }

  /**
   * Verifies the core Rotation contract for any implementation: round-trip, distance preservation,
   * norm preservation, and determinism.
   */
  private static void verifyRotationContract(Rotation rot, int dim) {
    Random rng = new Random(123L);

    // Round-trip: inverseRotate(rotate(v)) ≈ v
    // Dense rotation at d=64 accumulates ~64 FMA rounding errors per element, so use 1e-2f
    for (int trial = 0; trial < 5; trial++) {
      float[] v = randomVector(dim, rng);
      float[] rotated = rot.rotate(v);
      float[] recovered = rot.inverseRotate(rotated);
      for (int d = 0; d < dim; d++) {
        assertThat(recovered[d]).isCloseTo(v[d], within(1e-2f));
      }
    }

    // Norm preservation: ||R*v|| ≈ ||v||
    for (int trial = 0; trial < 5; trial++) {
      float[] v = randomVector(dim, rng);
      float normBefore = norm(v);
      float normAfter = norm(rot.rotate(v));
      assertThat(normAfter).isCloseTo(normBefore, within(1e-3f));
    }

    // Distance preservation: ||R*a - R*b|| ≈ ||a - b||
    for (int trial = 0; trial < 5; trial++) {
      float[] a = randomVector(dim, rng);
      float[] b = randomVector(dim, rng);
      float[] ra = rot.rotate(a);
      float[] rb = rot.rotate(b);

      float distBefore = (float) Math.sqrt(VectorUtil.squareDistance(a, b));
      float distAfter = (float) Math.sqrt(VectorUtil.squareDistance(ra, rb));
      assertThat(distAfter).isCloseTo(distBefore, within(1e-2f));
    }

    // Dot product preservation: <R*a, R*b> ≈ <a, b>
    for (int trial = 0; trial < 5; trial++) {
      float[] a = randomVector(dim, rng);
      float[] b = randomVector(dim, rng);
      float[] ra = rot.rotate(a);
      float[] rb = rot.rotate(b);

      float dotBefore = VectorUtil.dotProduct(a, b);
      float dotAfter = VectorUtil.dotProduct(ra, rb);
      assertThat(dotAfter).isCloseTo(dotBefore, within(Math.abs(dotBefore) * 0.01f + 0.01f));
    }
  }

  // --- GivensRotation Tests ---

  @Nested
  @Tag("unit")
  class GivensRotationTests {

    @Test
    void roundTrip_recoversOriginal() {
      int dim = 128;
      GivensRotation rot = GivensRotation.generate(dim, 42L);
      verifyRotationContract(rot, dim);
    }

    @Test
    void preservesDistances_128d() {
      int dim = 128;
      GivensRotation rot = GivensRotation.generate(dim, 42L);
      Random rng = new Random(99L);
      for (int trial = 0; trial < 10; trial++) {
        float[] a = randomVector(dim, rng);
        float[] b = randomVector(dim, rng);
        float distBefore = (float) Math.sqrt(VectorUtil.squareDistance(a, b));
        float distAfter =
            (float) Math.sqrt(VectorUtil.squareDistance(rot.rotate(a), rot.rotate(b)));
        assertThat(distAfter).isCloseTo(distBefore, within(1e-2f));
      }
    }

    @Test
    void deterministic_withSameSeed() {
      int dim = 64;
      GivensRotation rot1 = GivensRotation.generate(dim, 42L);
      GivensRotation rot2 = GivensRotation.generate(dim, 42L);
      float[] v = randomVector(dim, new Random(1L));
      float[] r1 = rot1.rotate(v);
      float[] r2 = rot2.rotate(v);
      for (int d = 0; d < dim; d++) {
        assertThat(r1[d]).isEqualTo(r2[d]);
      }
    }

    @Test
    void different_withDifferentSeeds() {
      int dim = 64;
      GivensRotation rot1 = GivensRotation.generate(dim, 42L);
      GivensRotation rot2 = GivensRotation.generate(dim, 99L);
      float[] v = randomVector(dim, new Random(1L));
      float[] r1 = rot1.rotate(v);
      float[] r2 = rot2.rotate(v);
      boolean anyDifferent = false;
      for (int d = 0; d < dim; d++) {
        if (Math.abs(r1[d] - r2[d]) > 1e-6f) {
          anyDifferent = true;
          break;
        }
      }
      assertThat(anyDifferent).isTrue();
    }

    @Test
    void oddDimension_lastElementPassesThrough() {
      int dim = 65;
      GivensRotation rot = GivensRotation.generate(dim, 42L);
      float[] v = randomVector(dim, new Random(1L));
      float[] rotated = rot.rotate(v);
      // Last element should be unchanged
      assertThat(rotated[64]).isEqualTo(v[64]);
      // But first elements should change
      boolean anyChanged = false;
      for (int d = 0; d < 64; d++) {
        if (Math.abs(rotated[d] - v[d]) > 1e-6f) {
          anyChanged = true;
          break;
        }
      }
      assertThat(anyChanged).isTrue();
    }

    @Test
    void dimensionMismatch_throws() {
      GivensRotation rot = GivensRotation.generate(64, 42L);
      assertThatThrownBy(() -> rot.rotate(new float[32]))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> rot.inverseRotate(new float[128]))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void dimensionTooSmall_throws() {
      assertThatThrownBy(() -> GivensRotation.generate(1, 42L))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> GivensRotation.generate(0, 42L))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void dimension_returnsCorrectValue() {
      GivensRotation rot = GivensRotation.generate(128, 42L);
      assertThat(rot.dimension()).isEqualTo(128);
    }

    @Test
    void worksWithRaBitQuantizer() {
      int dim = 32;
      int paddedDim = ((dim + 63) / 64) * 64; // 64
      Random rng = new Random(42L);
      float[][] vectors = new float[100][dim];
      for (int i = 0; i < 100; i++) {
        for (int d = 0; d < dim; d++) {
          vectors[i][d] = (float) rng.nextGaussian();
        }
      }
      VectorDataset dataset = new ArrayVectorDataset(vectors);
      Rotation rotation = GivensRotation.generate(paddedDim, 42L);
      RaBitQuantizer rq = RaBitQuantizer.train(dataset, rotation);
      RaBitQuantizedVectors compressed = rq.encodeAll(dataset);
      assertThat(compressed.size()).isEqualTo(100);

      // Score function should work
      float[] query = vectors[0];
      ScoreFunction scorer =
          compressed.scoreFunctionFor(
              query, com.integrallis.vectors.core.SimilarityFunction.EUCLIDEAN);
      float score = scorer.score(0);
      assertThat(score).isGreaterThan(0f).isLessThanOrEqualTo(1f);
    }
  }

  // --- QuaternionRotation Tests ---

  @Nested
  @Tag("unit")
  class QuaternionRotationTests {

    @Test
    void roundTrip_recoversOriginal() {
      int dim = 128;
      QuaternionRotation rot = QuaternionRotation.generate(dim, 42L);
      verifyRotationContract(rot, dim);
    }

    @Test
    void preservesDistances_128d() {
      int dim = 128;
      QuaternionRotation rot = QuaternionRotation.generate(dim, 42L);
      Random rng = new Random(99L);
      for (int trial = 0; trial < 10; trial++) {
        float[] a = randomVector(dim, rng);
        float[] b = randomVector(dim, rng);
        float distBefore = (float) Math.sqrt(VectorUtil.squareDistance(a, b));
        float distAfter =
            (float) Math.sqrt(VectorUtil.squareDistance(rot.rotate(a), rot.rotate(b)));
        assertThat(distAfter).isCloseTo(distBefore, within(1e-2f));
      }
    }

    @Test
    void deterministic_withSameSeed() {
      int dim = 128;
      QuaternionRotation rot1 = QuaternionRotation.generate(dim, 42L);
      QuaternionRotation rot2 = QuaternionRotation.generate(dim, 42L);
      float[] v = randomVector(dim, new Random(1L));
      float[] r1 = rot1.rotate(v);
      float[] r2 = rot2.rotate(v);
      for (int d = 0; d < dim; d++) {
        assertThat(r1[d]).isEqualTo(r2[d]);
      }
    }

    @Test
    void different_withDifferentSeeds() {
      int dim = 128;
      QuaternionRotation rot1 = QuaternionRotation.generate(dim, 42L);
      QuaternionRotation rot2 = QuaternionRotation.generate(dim, 99L);
      float[] v = randomVector(dim, new Random(1L));
      float[] r1 = rot1.rotate(v);
      float[] r2 = rot2.rotate(v);
      boolean anyDifferent = false;
      for (int d = 0; d < dim; d++) {
        if (Math.abs(r1[d] - r2[d]) > 1e-6f) {
          anyDifferent = true;
          break;
        }
      }
      assertThat(anyDifferent).isTrue();
    }

    @Test
    void nonMultipleOf4_trailingElementsPassThrough() {
      int dim = 130; // 32 full blocks + 2 trailing
      QuaternionRotation rot = QuaternionRotation.generate(dim, 42L);
      float[] v = randomVector(dim, new Random(1L));
      float[] rotated = rot.rotate(v);
      // Trailing elements should be unchanged
      assertThat(rotated[128]).isEqualTo(v[128]);
      assertThat(rotated[129]).isEqualTo(v[129]);
    }

    @Test
    void dimensionMismatch_throws() {
      QuaternionRotation rot = QuaternionRotation.generate(128, 42L);
      assertThatThrownBy(() -> rot.rotate(new float[64]))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> rot.inverseRotate(new float[256]))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void dimensionTooSmall_throws() {
      assertThatThrownBy(() -> QuaternionRotation.generate(3, 42L))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> QuaternionRotation.generate(0, 42L))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void dimension_returnsCorrectValue() {
      QuaternionRotation rot = QuaternionRotation.generate(128, 42L);
      assertThat(rot.dimension()).isEqualTo(128);
    }

    @Test
    void worksWithRaBitQuantizer() {
      int dim = 32;
      int paddedDim = ((dim + 63) / 64) * 64; // 64
      Random rng = new Random(42L);
      float[][] vectors = new float[100][dim];
      for (int i = 0; i < 100; i++) {
        for (int d = 0; d < dim; d++) {
          vectors[i][d] = (float) rng.nextGaussian();
        }
      }
      VectorDataset dataset = new ArrayVectorDataset(vectors);
      Rotation rotation = QuaternionRotation.generate(paddedDim, 42L);
      RaBitQuantizer rq = RaBitQuantizer.train(dataset, rotation);
      RaBitQuantizedVectors compressed = rq.encodeAll(dataset);
      assertThat(compressed.size()).isEqualTo(100);

      float[] query = vectors[0];
      ScoreFunction scorer =
          compressed.scoreFunctionFor(
              query, com.integrallis.vectors.core.SimilarityFunction.EUCLIDEAN);
      float score = scorer.score(0);
      assertThat(score).isGreaterThan(0f).isLessThanOrEqualTo(1f);
    }
  }

  // --- Rotation interface polymorphism ---

  @Nested
  @Tag("unit")
  class RotationInterfaceTests {

    @Test
    void allImplementations_satisfyContract_64d() {
      int dim = 64;
      Rotation dense = RandomRotation.generate(dim, 42L);
      Rotation givens = GivensRotation.generate(dim, 42L);
      Rotation quaternion = QuaternionRotation.generate(dim, 42L);

      verifyRotationContract(dense, dim);
      verifyRotationContract(givens, dim);
      verifyRotationContract(quaternion, dim);
    }

    @Test
    void allImplementations_satisfyContract_128d() {
      int dim = 128;
      Rotation dense = RandomRotation.generate(dim, 42L);
      Rotation givens = GivensRotation.generate(dim, 42L);
      Rotation quaternion = QuaternionRotation.generate(dim, 42L);

      verifyRotationContract(dense, dim);
      verifyRotationContract(givens, dim);
      verifyRotationContract(quaternion, dim);
    }

    @Test
    void randomRotation_dimensionTooSmall_throws() {
      assertThatThrownBy(() -> RandomRotation.generate(0, 42L))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> RandomRotation.generate(-1, 42L))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rotationIsPolymorphic_canBeUsedInterchangeably() {
      int dim = 64;
      float[] v = randomVector(dim, new Random(1L));

      Rotation[] rotations = {
        RandomRotation.generate(dim, 42L),
        GivensRotation.generate(dim, 42L),
        QuaternionRotation.generate(dim, 42L)
      };

      for (Rotation rot : rotations) {
        assertThat(rot.dimension()).isEqualTo(dim);
        float[] rotated = rot.rotate(v);
        assertThat(rotated).hasSize(dim);
        float[] recovered = rot.inverseRotate(rotated);
        assertThat(recovered).hasSize(dim);
        // Verify round-trip
        float normDiff = 0f;
        for (int d = 0; d < dim; d++) {
          float diff = recovered[d] - v[d];
          normDiff += diff * diff;
        }
        assertThat(normDiff).isLessThan(0.01f);
      }
    }
  }
}
