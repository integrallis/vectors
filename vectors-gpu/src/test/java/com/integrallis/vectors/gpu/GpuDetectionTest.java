/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.gpu;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Verifies that GPU detection never throws and produces consistent {@link GpuAvailability} results
 * on hosts without {@code libcuvs.so} installed. Actual GPU-backed search behaviour is covered by
 * {@code @Tag("slow")} tests that only run in CI hosts with CUDA drivers.
 */
class GpuDetectionTest {

  @Test
  @Tag("unit")
  void cuvsBackend_reportsUnavailable_whenNativeLibraryMissing() {
    var backend = new CuVsGpuBackend();
    assertThat(backend.name()).isEqualTo("cuvs");

    var availability = backend.detect();

    // CI hosts without CUDA must not crash; hosts with CUDA are free to report available.
    if (!availability.isAvailable()) {
      assertThat(availability.reason()).isPresent();
      assertThat(availability.providerName()).isEmpty();
      assertThat(availability.deviceCount()).isZero();
    } else {
      assertThat(availability.providerName()).contains("cuvs");
      assertThat(availability.deviceCount()).isPositive();
    }
  }

  @Test
  @Tag("unit")
  void gpuProvider_exposesSameAvailabilityAsBackend() {
    var availability = GpuProvider.availability();
    assertThat(availability).isNotNull();
    if (availability.isAvailable()) {
      assertThat(GpuProvider.backend()).isNotNull();
    } else {
      assertThat(GpuProvider.backend()).isNull();
      assertThat(availability.reason()).isPresent();
    }
  }

  @Test
  @Tag("unit")
  void availabilityFactoryMethods_roundTrip() {
    var a = GpuAvailability.available("cuvs", 2);
    assertThat(a.isAvailable()).isTrue();
    assertThat(a.providerName()).contains("cuvs");
    assertThat(a.deviceCount()).isEqualTo(2);
    assertThat(a.reason()).isEmpty();

    var u = GpuAvailability.unavailable("no driver");
    assertThat(u.isAvailable()).isFalse();
    assertThat(u.providerName()).isEmpty();
    assertThat(u.deviceCount()).isZero();
    assertThat(u.reason()).contains("no driver");
  }
}
