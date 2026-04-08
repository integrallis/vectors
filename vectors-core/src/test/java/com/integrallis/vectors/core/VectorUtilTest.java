package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for the {@link VectorUtil} public API facade. */
@Tag("unit")
class VectorUtilTest {

  @Test
  void dotProduct_knownValues() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 5.0f, 6.0f};
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assertThat(VectorUtil.dotProduct(a, b)).isCloseTo(32.0f, within(1e-4f));
  }

  @Test
  void squareDistance_knownValues() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 6.0f, 8.0f};
    // (1-4)^2 + (2-6)^2 + (3-8)^2 = 9 + 16 + 25 = 50
    assertThat(VectorUtil.squareDistance(a, b)).isCloseTo(50.0f, within(1e-4f));
  }

  @Test
  void cosine_parallelVectors() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {2.0f, 4.0f, 6.0f};
    assertThat(VectorUtil.cosine(a, b)).isCloseTo(1.0f, within(1e-4f));
  }

  @Test
  void cosine_orthogonalVectors() {
    float[] a = {1.0f, 0.0f};
    float[] b = {0.0f, 1.0f};
    assertThat(VectorUtil.cosine(a, b)).isCloseTo(0.0f, within(1e-4f));
  }

  @Test
  void dotProduct_dimensionMismatchThrows() {
    float[] a = {1.0f, 2.0f};
    float[] b = {1.0f, 2.0f, 3.0f};
    assertThatThrownBy(() -> VectorUtil.dotProduct(a, b))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimensions differ");
  }

  @Test
  void squareDistance_dimensionMismatchThrows() {
    float[] a = {1.0f};
    float[] b = {1.0f, 2.0f};
    assertThatThrownBy(() -> VectorUtil.squareDistance(a, b))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void byteDotProduct_knownValues() {
    byte[] a = {1, 2, 3, 4};
    byte[] b = {5, 6, 7, 8};
    assertThat(VectorUtil.dotProduct(a, b)).isEqualTo(70);
  }

  @Test
  void hammingDistance_knownValues() {
    long[] a = {0L, 0L};
    long[] b = {0xFFL, 0L}; // 8 bits differ
    assertThat(VectorUtil.hammingDistance(a, b)).isEqualTo(8);
  }

  @Test
  void addInPlace_knownValues() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {10.0f, 20.0f, 30.0f};
    VectorUtil.addInPlace(a, b);
    assertThat(a).containsExactly(11.0f, 22.0f, 33.0f);
  }

  @Test
  void subInPlace_knownValues() {
    float[] a = {10.0f, 20.0f, 30.0f};
    float[] b = {1.0f, 2.0f, 3.0f};
    VectorUtil.subInPlace(a, b);
    assertThat(a).containsExactly(9.0f, 18.0f, 27.0f);
  }

  @Test
  void scale_knownValues() {
    float[] a = {1.0f, 2.0f, 3.0f};
    VectorUtil.scale(a, 2.0f);
    assertThat(a).containsExactly(2.0f, 4.0f, 6.0f);
  }

  @Test
  void sum_knownValues() {
    float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
    assertThat(VectorUtil.sum(a)).isCloseTo(10.0f, within(1e-5f));
  }

  @Test
  void l2normalize_knownValues() {
    float[] v = {3.0f, 4.0f};
    VectorUtil.l2normalize(v, true);
    // ||[3,4]|| = 5, so normalized = [0.6, 0.8]
    assertThat(v[0]).isCloseTo(0.6f, within(1e-4f));
    assertThat(v[1]).isCloseTo(0.8f, within(1e-4f));
  }

  @Test
  void dotProduct_withOffset() {
    float[] a = {0.0f, 1.0f, 2.0f, 3.0f, 0.0f};
    float[] b = {0.0f, 4.0f, 5.0f, 6.0f, 0.0f};
    // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
    assertThat(VectorUtil.dotProduct(a, 1, b, 1, 3)).isCloseTo(32.0f, within(1e-4f));
  }

  @Test
  void cosine_memorySegment_parallelVectors() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {2.0f, 4.0f, 6.0f};
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segA = arena.allocate(ValueLayout.JAVA_FLOAT, a.length);
      MemorySegment segB = arena.allocate(ValueLayout.JAVA_FLOAT, b.length);
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_FLOAT, 0, a.length);
      MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_FLOAT, 0, b.length);
      assertThat(VectorUtil.cosine(segA, segB, a.length)).isCloseTo(1.0f, within(1e-4f));
    }
  }

  @Test
  void cosine_memorySegment_matchesArrayVersion() {
    float[] a = {1.0f, 0.5f, -0.5f, 0.3f};
    float[] b = {0.2f, 0.8f, 0.1f, -0.4f};
    float expected = VectorUtil.cosine(a, b);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segA = arena.allocate(ValueLayout.JAVA_FLOAT, a.length);
      MemorySegment segB = arena.allocate(ValueLayout.JAVA_FLOAT, b.length);
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_FLOAT, 0, a.length);
      MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_FLOAT, 0, b.length);
      assertThat(VectorUtil.cosine(segA, segB, a.length)).isCloseTo(expected, within(1e-4f));
    }
  }
}
