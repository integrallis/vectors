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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Tests for {@link VectorizationProvider}. */
@Tag("unit")
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
