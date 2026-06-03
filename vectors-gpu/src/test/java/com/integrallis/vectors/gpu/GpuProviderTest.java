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

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GpuProviderTest {

  @Test
  void resolveReturnsSelectedBackendWithoutSecondDetectCall() {
    FakeBackend unavailable = new FakeBackend("missing", GpuAvailability.unavailable("missing"));
    FakeBackend selected = new FakeBackend("selected", GpuAvailability.available("selected", 1));

    GpuProvider.Resolved resolved = GpuProvider.resolve(List.of(unavailable, selected));

    assertThat(resolved.availability().isAvailable()).isTrue();
    assertThat(resolved.backend()).isSameAs(selected);
    assertThat(unavailable.detectCalls).isEqualTo(1);
    assertThat(selected.detectCalls).isEqualTo(1);
  }

  @Test
  void resolveReturnsUnavailableWhenNoBackendMatches() {
    FakeBackend first = new FakeBackend("first", GpuAvailability.unavailable("first missing"));
    FakeBackend second = new FakeBackend("second", GpuAvailability.unavailable("second missing"));

    GpuProvider.Resolved resolved = GpuProvider.resolve(List.of(first, second));

    assertThat(resolved.availability().isAvailable()).isFalse();
    assertThat(resolved.availability().reason())
        .contains("no GpuBackend service advertised compatibility");
    assertThat(resolved.backend()).isNull();
    assertThat(first.detectCalls).isEqualTo(1);
    assertThat(second.detectCalls).isEqualTo(1);
  }

  private static final class FakeBackend implements GpuBackend {
    private final String name;
    private final GpuAvailability availability;
    private int detectCalls;

    private FakeBackend(String name, GpuAvailability availability) {
      this.name = name;
      this.availability = availability;
    }

    @Override
    public String name() {
      return name;
    }

    @Override
    public GpuAvailability detect() {
      detectCalls++;
      return availability;
    }
  }
}
