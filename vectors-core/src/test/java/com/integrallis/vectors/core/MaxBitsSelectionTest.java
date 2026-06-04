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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins the SIMD register-width cap (P1.5). The default cap matches Lucene's {@code
 * MAX_BITS_PER_VECTOR=256} to avoid AVX-512 frequency downclock; {@code -Dvectors.maxBits=512} opts
 * back into wider registers on hardware that does not downclock.
 */
@Tag("unit")
class MaxBitsSelectionTest {

  @Test
  void defaultCapIs256() {
    assertThat(PanamaConstants.DEFAULT_MAX_BITS).isEqualTo(256);
  }

  @Test
  void resolvedCapDefaultsTo256WhenPropertyUnset() {
    // The test JVM does not set -Dvectors.maxBits, so the resolved cap is the default.
    assertThat(System.getProperty("vectors.maxBits")).isNull();
    assertThat(PanamaConstants.MAX_BITS).isEqualTo(PanamaConstants.DEFAULT_MAX_BITS);
  }

  @Test
  void parsesValidPowerOfTwoWidths() {
    assertThat(PanamaConstants.parseMaxBits("64")).isEqualTo(64);
    assertThat(PanamaConstants.parseMaxBits("128")).isEqualTo(128);
    assertThat(PanamaConstants.parseMaxBits("256")).isEqualTo(256);
    assertThat(PanamaConstants.parseMaxBits("512")).isEqualTo(512);
  }

  @Test
  void rejectsNonPowerOfTwo() {
    assertThatThrownBy(() -> PanamaConstants.parseMaxBits("200"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("power of two");
  }

  @Test
  void rejectsOutOfRangeWidths() {
    assertThatThrownBy(() -> PanamaConstants.parseMaxBits("32"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> PanamaConstants.parseMaxBits("1024"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNonNumericValue() {
    assertThatThrownBy(() -> PanamaConstants.parseMaxBits("wide"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void capsSpeciesWiderThanTheCeiling() {
    VectorSpecies<Float> capped = PanamaConstants.preferredSpecies(FloatVector.SPECIES_512);
    assertThat(capped.vectorBitSize()).isEqualTo(Math.min(512, PanamaConstants.MAX_BITS));
    assertThat(capped.elementType()).isEqualTo(float.class);
  }

  @Test
  void leavesSpeciesWithinTheCeilingUntouched() {
    VectorSpecies<Float> within = PanamaConstants.preferredSpecies(FloatVector.SPECIES_64);
    assertThat(within.vectorBitSize()).isEqualTo(64);
    assertThat(within).isSameAs(FloatVector.SPECIES_64);
  }

  @Test
  void effectiveKernelWidthNeverExceedsCapOrHardware() {
    assertThat(PanamaVectorUtilSupport.VECTOR_BITSIZE)
        .isLessThanOrEqualTo(PanamaConstants.MAX_BITS)
        .isLessThanOrEqualTo(PanamaConstants.PREFERRED_BITS);
  }
}
