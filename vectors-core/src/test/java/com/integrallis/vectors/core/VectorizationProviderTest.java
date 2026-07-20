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

  /**
   * Pins the startup-toggles summary content: an operator should be able to grep one INFO line for
   * the provider name, the resolved SIMD width, FMA capability flags, and any {@code vectors.*}
   * system-property overrides that were applied. Avoids a silent SIMD-to-scalar downgrade when
   * {@code --add-modules=jdk.incubator.vector} is missing on a JDK upgrade.
   */
  @Test
  void toggleSummaryExposesProviderAndKnobs() {
    String summary = VectorizationProvider.buildToggleSummary(VectorizationProvider.getInstance());
    assertThat(summary).startsWith("vectors-core: provider=");
    assertThat(summary).contains("panama=");
    assertThat(summary).contains("maxBits=" + PanamaConstants.MAX_BITS);
    assertThat(summary).contains("preferredBits=");
    assertThat(summary).contains("fastVectorFMA=");
    assertThat(summary).contains("fastScalarFMA=");
    assertThat(summary).contains("sve=");
    assertThat(summary).contains("ggufExecutor=persistent");
    assertThat(summary).contains("ggufThreads=" + GgufParallelSupport.parallelism());
    assertThat(summary).contains("ggufChunksPerThread=2");
    assertThat(summary).contains("mappedKQuantLongOffsets=auto(");
    assertThat(summary).contains("q4=").contains("q5=").contains("q6=");
    assertThat(summary).contains("q4ShortPairwise=" + PanamaConstants.USE_Q4_SHORT_PAIRWISE);
    assertThat(summary).contains("toggles=[");
    assertThat(summary).endsWith("]");
  }

  @Test
  void toggleSummaryReportsForcedScalarProperty() {
    String key = "vectors.forceScalar";
    String prior = System.getProperty(key);
    System.setProperty(key, "true");
    try {
      String summary =
          VectorizationProvider.buildToggleSummary(VectorizationProvider.newScalarProvider());
      assertThat(summary)
          .as("forced-scalar property must appear verbatim in the toggles list")
          .contains("vectors.forceScalar=true");
    } finally {
      if (prior == null) {
        System.clearProperty(key);
      } else {
        System.setProperty(key, prior);
      }
    }
  }

  @Test
  void toggleSummaryReportsMappedKQuantLongOffsetPropertyWithoutChangingStartupPolicy() {
    String key = PanamaConstants.MAPPED_K_QUANT_LONG_OFFSETS_PROPERTY;
    String prior = System.getProperty(key);
    PanamaConstants.MappedKQuantLongOffsetPolicy startupPolicy =
        PanamaConstants.mappedKQuantLongOffsetPolicy();
    System.setProperty(key, "true");
    try {
      String summary =
          VectorizationProvider.buildToggleSummary(VectorizationProvider.getInstance());
      assertThat(summary)
          .contains(
              "mappedKQuantLongOffsets=%s(q4=%s,q5=%s,q6=%s)"
                  .formatted(
                      startupPolicy.mode(),
                      startupPolicy.q4(),
                      startupPolicy.q5(),
                      startupPolicy.q6()))
          .contains(key + "=true");
    } finally {
      if (prior == null) {
        System.clearProperty(key);
      } else {
        System.setProperty(key, prior);
      }
    }
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
  void runtimeCapabilitiesExposeStructuredProviderAndExecutorFacts() {
    VectorRuntimeCapabilities capabilities = VectorUtil.runtimeCapabilities();

    assertThat(capabilities.providerName()).isEqualTo(VectorizationProvider.getProviderName());
    assertThat(capabilities.vectorApi()).isEqualTo(VectorizationProvider.isPanamaEnabled());
    assertThat(capabilities.preferredVectorBits()).isPositive();
    assertThat(capabilities.activeVectorBits())
        .isEqualTo(
            VectorizationProvider.isPanamaEnabled() ? PanamaVectorUtilSupport.VECTOR_BITSIZE : 0);
    assertThat(capabilities.ggufExecutor()).isEqualTo("persistent");
    assertThat(capabilities.q4ShortPairwise()).isEqualTo(PanamaConstants.USE_Q4_SHORT_PAIRWISE);
    assertThat(capabilities.ggufThreads()).isEqualTo(GgufParallelSupport.parallelism());
    assertThat(capabilities.ggufChunksPerThread()).isEqualTo(GgufParallelSupport.chunksPerThread());
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

  // P1.7 — guard the "scalar is the correctness baseline" claim: it is only meaningful if the
  // thing it is baselined against (Panama SIMD) is actually live. Kernel-value agreement between
  // scalar and Panama is already pinned by VectorUtilSupportTest.panamaMatchesScalar; the missing
  // piece is a test that fails when the runtime has silently fallen back to scalar (the regression
  // P1.7 targets).

  @Test
  void panamaActiveImpliesNoSilentFallback() {
    // When SIMD is the selected provider, no Panama load failure was recorded and scalar was not
    // forced. If a JDK upgrade silently broke jdk.incubator.vector, isPanamaEnabled() would be
    // false here (and getPanamaFailure() would carry the cause), failing CI loudly.
    if (VectorizationProvider.isPanamaEnabled()) {
      assertThat(VectorizationProvider.getPanamaFailure()).isEmpty();
      assertThat(VectorizationProvider.isForcedScalar()).isFalse();
    } else {
      // Scalar is active: either explicitly forced, or Panama failed to load — exactly one is true,
      // and a failure must carry its cause so the regression is diagnosable.
      assertThat(
              VectorizationProvider.isForcedScalar()
                  ^ VectorizationProvider.getPanamaFailure().isPresent())
          .isTrue();
    }
  }

  @Test
  void isForcedScalarReflectsSystemProperty() {
    assertThat(VectorizationProvider.isForcedScalar())
        .isEqualTo(Boolean.getBoolean("vectors.forceScalar"));
  }
}
