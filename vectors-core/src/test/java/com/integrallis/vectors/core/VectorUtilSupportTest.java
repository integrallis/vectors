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
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Comprehensive tests for {@link VectorUtilSupport} implementations. Tests run against both scalar
 * and Panama providers to ensure cross-implementation agreement.
 */
@Tag("unit")
class VectorUtilSupportTest {

  /** Relative epsilon for float comparisons (1e-5 as specified in the plan). */
  private static final float FLOAT_EPSILON = 1e-4f;

  private static final long SEED = 42L;

  private VectorUtilSupport scalar;
  private VectorUtilSupport panama;

  @BeforeEach
  void setUp() {
    scalar = VectorizationProvider.newScalarProvider();
    panama = VectorizationProvider.getInstance();
  }

  // --- Test dimensions covering all tail/alignment scenarios ---

  static int[] testDimensions() {
    return new int[] {
      1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300,
      384, 511, 512, 513, 768, 1023, 1024, 1025, 1536
    };
  }

  static Stream<Arguments> dimensionProvider() {
    return java.util.Arrays.stream(testDimensions()).mapToObj(Arguments::of);
  }

  // --- Helper methods ---

  private static float[] randomFloats(int dim, Random rng) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2f - 1f; // [-1, 1]
    }
    return v;
  }

  private static byte[] randomBytes(int dim, Random rng) {
    byte[] v = new byte[dim];
    rng.nextBytes(v);
    return v;
  }

  private static long[] randomLongs(int dim, Random rng) {
    long[] v = new long[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextLong();
    }
    return v;
  }

  /** Naive (non-optimized) dot product for correctness baseline. */
  private static float naiveDotProduct(float[] a, float[] b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      sum += (double) a[i] * (double) b[i];
    }
    return (float) sum;
  }

  /** Naive squared L2 distance. */
  private static float naiveSquareDistance(float[] a, float[] b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      double diff = a[i] - b[i];
      sum += diff * diff;
    }
    return (float) sum;
  }

  /** Naive cosine similarity. */
  private static float naiveCosine(float[] a, float[] b) {
    double sum = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for (int i = 0; i < a.length; i++) {
      sum += (double) a[i] * (double) b[i];
      norm1 += (double) a[i] * (double) a[i];
      norm2 += (double) b[i] * (double) b[i];
    }
    return (float) (sum / Math.sqrt(norm1 * norm2));
  }

  // ===========================
  // Float dot product tests
  // ===========================

  @Nested
  class FloatDotProductTest {

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void matchesNaive(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float expected = naiveDotProduct(a, b);

      assertThat(scalar.dotProduct(a, b)).isCloseTo(expected, within(FLOAT_EPSILON * dim));
      assertThat(panama.dotProduct(a, b)).isCloseTo(expected, within(FLOAT_EPSILON * dim));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void panamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);

      float scalarResult = scalar.dotProduct(a, b);
      float panamaResult = panama.dotProduct(a, b);

      assertThat(panamaResult).isCloseTo(scalarResult, within(FLOAT_EPSILON * dim));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void selfDotProductIsSquaredNorm(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);

      float dot = panama.dotProduct(a, a);
      assertThat(dot).isGreaterThanOrEqualTo(0f);
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void withOffsetMatchesFull(int dim) {
      if (dim < 4) return;
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      int offset = 2;
      int len = dim - 4;

      float[] aSub = new float[len];
      float[] bSub = new float[len];
      System.arraycopy(a, offset, aSub, 0, len);
      System.arraycopy(b, offset, bSub, 0, len);

      float expected = panama.dotProduct(aSub, bSub);
      float actual = panama.dotProduct(a, offset, b, offset, len);

      assertThat(actual).isCloseTo(expected, within(FLOAT_EPSILON * len));
    }

    @Test
    void emptyVectors() {
      float[] a = {};
      float[] b = {};
      assertThat(panama.dotProduct(a, b)).isEqualTo(0f);
    }

    @Test
    void singleElement() {
      float[] a = {3.0f};
      float[] b = {4.0f};
      assertThat(panama.dotProduct(a, b)).isCloseTo(12.0f, within(FLOAT_EPSILON));
    }
  }

  // ===========================
  // Float square distance tests
  // ===========================

  @Nested
  class FloatSquareDistanceTest {

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void matchesNaive(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float expected = naiveSquareDistance(a, b);

      assertThat(scalar.squareDistance(a, b)).isCloseTo(expected, within(FLOAT_EPSILON * dim));
      assertThat(panama.squareDistance(a, b)).isCloseTo(expected, within(FLOAT_EPSILON * dim));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void panamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);

      assertThat(panama.squareDistance(a, b))
          .isCloseTo(scalar.squareDistance(a, b), within(FLOAT_EPSILON * dim));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void selfDistanceIsZero(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      assertThat(panama.squareDistance(a, a)).isCloseTo(0f, within(FLOAT_EPSILON));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void isNonNegative(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      assertThat(panama.squareDistance(a, b)).isGreaterThanOrEqualTo(0f);
    }

    @ParameterizedTest(name = "dim={0}")
    @ValueSource(ints = {3, 16, 64, 256, 768})
    void satisfiesTriangleInequality(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float[] c = randomFloats(dim, rng);

      double dab = Math.sqrt(panama.squareDistance(a, b));
      double dbc = Math.sqrt(panama.squareDistance(b, c));
      double dac = Math.sqrt(panama.squareDistance(a, c));

      // Triangle inequality: d(a,c) <= d(a,b) + d(b,c) (with small epsilon for float imprecision)
      assertThat(dac).isLessThanOrEqualTo(dab + dbc + 1e-3);
    }
  }

  // ===========================
  // Float cosine similarity tests
  // ===========================

  @Nested
  class FloatCosineTest {

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void matchesNaive(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float expected = naiveCosine(a, b);

      assertThat(scalar.cosine(a, b)).isCloseTo(expected, within(FLOAT_EPSILON));
      assertThat(panama.cosine(a, b)).isCloseTo(expected, within(FLOAT_EPSILON));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void panamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);

      assertThat(panama.cosine(a, b)).isCloseTo(scalar.cosine(a, b), within(FLOAT_EPSILON));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void selfCosineIsOne(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      assertThat(panama.cosine(a, a)).isCloseTo(1.0f, within(FLOAT_EPSILON));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void rangeIsMinus1To1(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);

      float cosine = panama.cosine(a, b);
      assertThat(cosine).isBetween(-1.0f - FLOAT_EPSILON, 1.0f + FLOAT_EPSILON);
    }

    @Test
    void oppositeVectorsIsMinusOne() {
      float[] a = {1.0f, 0.0f, 0.0f};
      float[] b = {-1.0f, 0.0f, 0.0f};
      assertThat(panama.cosine(a, b)).isCloseTo(-1.0f, within(FLOAT_EPSILON));
    }

    @Test
    void orthogonalVectorsIsZero() {
      float[] a = {1.0f, 0.0f};
      float[] b = {0.0f, 1.0f};
      assertThat(panama.cosine(a, b)).isCloseTo(0.0f, within(FLOAT_EPSILON));
    }
  }

  // ===========================
  // Byte distance tests
  // ===========================

  @Nested
  class ByteDistanceTest {

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void dotProductPanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      byte[] a = randomBytes(dim, rng);
      byte[] b = randomBytes(dim, rng);

      assertThat(panama.dotProduct(a, b)).isEqualTo(scalar.dotProduct(a, b));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void squareDistancePanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      byte[] a = randomBytes(dim, rng);
      byte[] b = randomBytes(dim, rng);

      assertThat(panama.squareDistance(a, b)).isEqualTo(scalar.squareDistance(a, b));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void cosinePanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      byte[] a = randomBytes(dim, rng);
      byte[] b = randomBytes(dim, rng);

      assertThat(panama.cosine(a, b)).isCloseTo(scalar.cosine(a, b), within(FLOAT_EPSILON));
    }

    @Test
    void byteDotProductKnownValues() {
      byte[] a = {1, 2, 3, 4};
      byte[] b = {5, 6, 7, 8};
      // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
      assertThat(panama.dotProduct(a, b)).isEqualTo(70);
      assertThat(scalar.dotProduct(a, b)).isEqualTo(70);
    }

    @Test
    void byteSquareDistanceKnownValues() {
      byte[] a = {1, 2, 3};
      byte[] b = {4, 6, 8};
      // (1-4)^2 + (2-6)^2 + (3-8)^2 = 9+16+25 = 50
      assertThat(panama.squareDistance(a, b)).isEqualTo(50);
      assertThat(scalar.squareDistance(a, b)).isEqualTo(50);
    }
  }

  // ===========================
  // Hamming distance tests
  // ===========================

  @Nested
  class HammingDistanceTest {

    @ParameterizedTest(name = "longs={0}")
    @ValueSource(ints = {1, 2, 3, 4, 7, 8, 15, 16, 32})
    void panamaMatchesScalar(int numLongs) {
      Random rng = new Random(SEED);
      long[] a = randomLongs(numLongs, rng);
      long[] b = randomLongs(numLongs, rng);

      assertThat(panama.hammingDistance(a, b)).isEqualTo(scalar.hammingDistance(a, b));
    }

    @Test
    void identicalVectorsIsZero() {
      long[] a = {0xDEADBEEFL, 0xCAFEBABEL};
      assertThat(panama.hammingDistance(a, a)).isEqualTo(0);
    }

    @Test
    void knownValues() {
      long[] a = {0L};
      long[] b = {0xFFL}; // 8 bits set
      assertThat(panama.hammingDistance(a, b)).isEqualTo(8);
    }

    @Test
    void allBitsFlipped() {
      long[] a = {0L};
      long[] b = {-1L}; // all 64 bits set
      assertThat(panama.hammingDistance(a, b)).isEqualTo(64);
    }
  }

  // ===========================
  // MemorySegment distance tests
  // ===========================

  @Nested
  class MemorySegmentTest {

    @ParameterizedTest(name = "dim={0}")
    @ValueSource(ints = {1, 8, 16, 32, 64, 128, 256, 768})
    void dotProductMatchesArrayVersion(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float expected = panama.dotProduct(a, b);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segA = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment segB = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_FLOAT, 0, dim);
        MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_FLOAT, 0, dim);

        float actual = panama.dotProduct(segA, segB, dim);
        assertThat(actual).isCloseTo(expected, within(FLOAT_EPSILON * dim));
      }
    }

    @ParameterizedTest(name = "dim={0}")
    @ValueSource(ints = {1, 8, 16, 32, 64, 128, 256, 768})
    void squareDistanceMatchesArrayVersion(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float expected = panama.squareDistance(a, b);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segA = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment segB = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_FLOAT, 0, dim);
        MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_FLOAT, 0, dim);

        float actual = panama.squareDistance(segA, segB, dim);
        assertThat(actual).isCloseTo(expected, within(FLOAT_EPSILON * dim));
      }
    }
  }

  // ===========================
  // MemorySegment cosine tests
  // ===========================

  @Nested
  class MemorySegmentCosineTest {

    @ParameterizedTest(name = "dim={0}")
    @ValueSource(ints = {1, 8, 16, 32, 64, 128, 256, 768})
    void cosineMatchesArrayVersion(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float expected = panama.cosine(a, b);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segA = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment segB = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_FLOAT, 0, dim);
        MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_FLOAT, 0, dim);

        float actual = panama.cosine(segA, segB, dim);
        assertThat(actual).isCloseTo(expected, within(FLOAT_EPSILON));
      }
    }

    @ParameterizedTest(name = "dim={0}")
    @ValueSource(ints = {1, 8, 16, 32, 64, 128, 256, 768})
    void panamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);

      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segA = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment segB = arena.allocate(ValueLayout.JAVA_FLOAT, dim);
        MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_FLOAT, 0, dim);
        MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_FLOAT, 0, dim);

        float scalarResult = scalar.cosine(segA, segB, dim);
        float panamaResult = panama.cosine(segA, segB, dim);
        assertThat(panamaResult).isCloseTo(scalarResult, within(FLOAT_EPSILON));
      }
    }

    @Test
    void selfCosineIsOne() {
      float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment seg = arena.allocate(ValueLayout.JAVA_FLOAT, a.length);
        MemorySegment.copy(a, 0, seg, ValueLayout.JAVA_FLOAT, 0, a.length);
        assertThat(panama.cosine(seg, seg, a.length)).isCloseTo(1.0f, within(FLOAT_EPSILON));
      }
    }

    @Test
    void orthogonalVectorsIsZero() {
      float[] a = {1.0f, 0.0f};
      float[] b = {0.0f, 1.0f};
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment segA = arena.allocate(ValueLayout.JAVA_FLOAT, 2);
        MemorySegment segB = arena.allocate(ValueLayout.JAVA_FLOAT, 2);
        MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_FLOAT, 0, 2);
        MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_FLOAT, 0, 2);
        assertThat(panama.cosine(segA, segB, 2)).isCloseTo(0.0f, within(FLOAT_EPSILON));
      }
    }
  }

  // ===========================
  // Vector arithmetic tests
  // ===========================

  @Nested
  class VectorArithmeticTest {

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void addInPlacePanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a1 = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float[] a2 = a1.clone();

      scalar.addInPlace(a1, b);
      panama.addInPlace(a2, b);

      for (int i = 0; i < dim; i++) {
        assertThat(a2[i]).as("element %d", i).isCloseTo(a1[i], within(FLOAT_EPSILON));
      }
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void addScaledInPlacePanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] out1 = randomFloats(dim + 2, rng);
      float[] vector = randomFloats(dim + 4, rng);
      float[] out2 = out1.clone();
      float multiplier = rng.nextFloat() * 4f - 2f;

      scalar.addScaledInPlace(out1, 1, vector, 2, dim, multiplier);
      panama.addScaledInPlace(out2, 1, vector, 2, dim, multiplier);

      for (int i = 0; i < out1.length; i++) {
        assertThat(out2[i]).as("element %d", i).isCloseTo(out1[i], within(FLOAT_EPSILON));
      }
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void subInPlacePanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a1 = randomFloats(dim, rng);
      float[] b = randomFloats(dim, rng);
      float[] a2 = a1.clone();

      scalar.subInPlace(a1, b);
      panama.subInPlace(a2, b);

      for (int i = 0; i < dim; i++) {
        assertThat(a2[i]).as("element %d", i).isCloseTo(a1[i], within(FLOAT_EPSILON));
      }
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void scalePanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a1 = randomFloats(dim, rng);
      float[] a2 = a1.clone();
      float multiplier = rng.nextFloat() * 4f - 2f;

      scalar.scale(a1, multiplier);
      panama.scale(a2, multiplier);

      for (int i = 0; i < dim; i++) {
        assertThat(a2[i]).as("element %d", i).isCloseTo(a1[i], within(FLOAT_EPSILON));
      }
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void sumPanamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] a = randomFloats(dim, rng);

      assertThat(panama.sum(a)).isCloseTo(scalar.sum(a), within(FLOAT_EPSILON * dim));
    }

    @Test
    void addInPlaceKnownValues() {
      float[] a = {1.0f, 2.0f, 3.0f};
      float[] b = {4.0f, 5.0f, 6.0f};
      panama.addInPlace(a, b);
      assertThat(a).containsExactly(5.0f, 7.0f, 9.0f);
    }

    @Test
    void subInPlaceKnownValues() {
      float[] a = {5.0f, 7.0f, 9.0f};
      float[] b = {4.0f, 5.0f, 6.0f};
      panama.subInPlace(a, b);
      assertThat(a).containsExactly(1.0f, 2.0f, 3.0f);
    }
  }

  // ===========================
  // L2 normalization tests
  // ===========================

  @Nested
  class L2NormalizeTest {

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void producesUnitVector(int dim) {
      Random rng = new Random(SEED);
      float[] v = randomFloats(dim, rng);

      panama.l2normalize(v, true);

      // Check that ||v|| ~= 1
      float norm = panama.dotProduct(v, v);
      assertThat(norm).isCloseTo(1.0f, within(1e-3f));
    }

    @ParameterizedTest(name = "dim={0}")
    @MethodSource("com.integrallis.vectors.core.VectorUtilSupportTest#dimensionProvider")
    void panamaMatchesScalar(int dim) {
      Random rng = new Random(SEED);
      float[] v1 = randomFloats(dim, rng);
      float[] v2 = v1.clone();

      scalar.l2normalize(v1, true);
      panama.l2normalize(v2, true);

      for (int i = 0; i < dim; i++) {
        assertThat(v2[i]).as("element %d", i).isCloseTo(v1[i], within(FLOAT_EPSILON));
      }
    }

    @Test
    void zeroVectorThrowsWhenRequested() {
      float[] v = {0.0f, 0.0f, 0.0f};
      org.junit.jupiter.api.Assertions.assertThrows(
          IllegalArgumentException.class, () -> panama.l2normalize(v, true));
    }

    @Test
    void zeroVectorReturnsZeroWhenNotThrowing() {
      float[] v = {0.0f, 0.0f, 0.0f};
      float[] result = panama.l2normalize(v, false);
      assertThat(result).containsExactly(0.0f, 0.0f, 0.0f);
    }

    @Test
    void alreadyUnitVectorNoChange() {
      float[] v = {1.0f, 0.0f, 0.0f};
      float[] original = v.clone();
      panama.l2normalize(v, true);
      assertThat(v).containsExactly(original);
    }
  }
}
