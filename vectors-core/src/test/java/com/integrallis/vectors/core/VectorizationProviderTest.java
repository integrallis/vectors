package com.integrallis.vectors.core;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/** Tests for {@link VectorizationProvider}. */
class VectorizationProviderTest {

  @Test
  void instanceIsNotNull() {
    assertThat(VectorizationProvider.getInstance()).isNotNull();
  }

  @Test
  void isPanamaEnabledReturnsBooleanWithoutCrashing() {
    // On JDK 25+ with --add-modules jdk.incubator.vector, this should be true
    boolean enabled = VectorizationProvider.isPanamaEnabled();
    assertThat(enabled).isNotNull();
  }

  @Test
  void providerNameIsNotEmpty() {
    String name = VectorizationProvider.getProviderName();
    assertThat(name).isNotEmpty();
    // Should be one of the known implementations
    assertThat(name).isIn("PanamaVectorUtilSupport", "ScalarVectorUtilSupport");
  }

  @Test
  void newScalarProviderReturnsScalar() {
    VectorUtilSupport scalar = VectorizationProvider.newScalarProvider();
    assertThat(scalar).isNotNull();
    assertThat(scalar.getClass().getSimpleName()).isEqualTo("ScalarVectorUtilSupport");
  }

  @Test
  void scalarProviderProducesCorrectResults() {
    VectorUtilSupport scalar = VectorizationProvider.newScalarProvider();
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 5.0f, 6.0f};
    assertThat(scalar.dotProduct(a, b)).isEqualTo(32.0f);
  }
}
