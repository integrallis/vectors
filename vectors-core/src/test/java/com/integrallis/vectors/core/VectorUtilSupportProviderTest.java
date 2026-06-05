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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests for the {@link VectorUtilSupportProvider} SPI and {@link VectorizationProvider#select}.
 *
 * <p>Distinct {@link VectorizationProvider#newScalarProvider()} instances serve as identity
 * sentinels for the chosen kernels — no mocking framework is needed (and Mockito cannot mock {@link
 * VectorUtilSupport} under the incubator Vector API on this JDK).
 */
@Tag("unit")
class VectorUtilSupportProviderTest {

  private static VectorUtilSupport sentinel() {
    return VectorizationProvider.newScalarProvider();
  }

  /** A synthetic provider whose availability, priority, and kernels are fully controlled. */
  private static final class FakeProvider implements VectorUtilSupportProvider {
    private final String name;
    private final int priority;
    private final boolean available;
    private final VectorUtilSupport impl;
    private final RuntimeException createError;

    FakeProvider(
        String name,
        int priority,
        boolean available,
        VectorUtilSupport impl,
        RuntimeException createError) {
      this.name = name;
      this.priority = priority;
      this.available = available;
      this.impl = impl;
      this.createError = createError;
    }

    static FakeProvider available(String name, int priority, VectorUtilSupport impl) {
      return new FakeProvider(name, priority, true, impl, null);
    }

    @Override
    public String name() {
      return name;
    }

    @Override
    public int priority() {
      return priority;
    }

    @Override
    public boolean isAvailable() {
      return available;
    }

    @Override
    public VectorUtilSupport create() {
      if (createError != null) {
        throw createError;
      }
      return impl;
    }
  }

  @Test
  void highestPriorityAvailableProviderWins() {
    VectorUtilSupport low = sentinel();
    VectorUtilSupport high = sentinel();
    List<VectorUtilSupportProvider> providers =
        List.of(FakeProvider.available("low", 10, low), FakeProvider.available("high", 200, high));

    assertThat(VectorizationProvider.select(providers)).isSameAs(high);
  }

  @Test
  void discoveryOrderBreaksPriorityTies() {
    VectorUtilSupport first = sentinel();
    VectorUtilSupport second = sentinel();
    List<VectorUtilSupportProvider> providers =
        List.of(
            FakeProvider.available("first", 50, first),
            FakeProvider.available("second", 50, second));

    // Stable sort keeps the earlier-discovered provider ahead at equal priority.
    assertThat(VectorizationProvider.select(providers)).isSameAs(first);
  }

  @Test
  void unavailableProvidersAreSkipped() {
    VectorUtilSupport fallback = sentinel();
    List<VectorUtilSupportProvider> providers =
        List.of(
            new FakeProvider("unavailable", 200, false, sentinel(), null),
            FakeProvider.available("usable", 10, fallback));

    assertThat(VectorizationProvider.select(providers)).isSameAs(fallback);
  }

  @Test
  void providerThatThrowsOnCreateIsSkipped() {
    VectorUtilSupport fallback = sentinel();
    List<VectorUtilSupportProvider> providers =
        List.of(
            new FakeProvider("boom", 200, true, null, new RuntimeException("create failed")),
            FakeProvider.available("usable", 10, fallback));

    assertThat(VectorizationProvider.select(providers)).isSameAs(fallback);
  }

  @Test
  void providerReturningNullIsSkipped() {
    VectorUtilSupport fallback = sentinel();
    List<VectorUtilSupportProvider> providers =
        List.of(
            FakeProvider.available("null-impl", 200, null),
            FakeProvider.available("usable", 10, fallback));

    assertThat(VectorizationProvider.select(providers)).isSameAs(fallback);
  }

  @Test
  void emptyProviderListFallsBackToScalarKernels() {
    VectorUtilSupport impl = VectorizationProvider.select(List.of());
    assertThat(impl).isInstanceOf(ScalarVectorUtilSupport.class);
  }

  @Test
  void allUnavailableFallsBackToScalarKernels() {
    List<VectorUtilSupportProvider> providers =
        List.of(new FakeProvider("nope", 200, false, sentinel(), null));
    assertThat(VectorizationProvider.select(providers)).isInstanceOf(ScalarVectorUtilSupport.class);
  }

  @Test
  void serviceLoaderDiscoversBothBuiltInProviders() {
    List<String> names = new ArrayList<>();
    int panamaPriority = Integer.MIN_VALUE;
    int scalarPriority = Integer.MIN_VALUE;
    for (VectorUtilSupportProvider provider :
        ServiceLoader.load(VectorUtilSupportProvider.class, getClass().getClassLoader())) {
      names.add(provider.name());
      if ("panama".equals(provider.name())) {
        panamaPriority = provider.priority();
      } else if ("scalar".equals(provider.name())) {
        scalarPriority = provider.priority();
      }
    }
    assertThat(names).contains("panama", "scalar");
    assertThat(panamaPriority).isEqualTo(VectorUtilSupportProvider.PANAMA_PRIORITY);
    assertThat(scalarPriority).isEqualTo(VectorUtilSupportProvider.SCALAR_PRIORITY);
    assertThat(panamaPriority).isGreaterThan(scalarPriority);
  }

  @Test
  void builtInScalarProviderIsAlwaysAvailable() {
    ScalarVectorUtilSupportProvider scalar = new ScalarVectorUtilSupportProvider();
    assertThat(scalar.isAvailable()).isTrue();
    assertThat(scalar.create()).isInstanceOf(ScalarVectorUtilSupport.class);
    assertThat(scalar.unavailabilityCause()).isEmpty();
  }

  @Test
  void builtInPanamaProviderReportsAvailabilityWithoutThrowing() {
    PanamaVectorUtilSupportProvider panama = new PanamaVectorUtilSupportProvider();
    boolean available = panama.isAvailable();
    if (available) {
      assertThat(panama.create()).isInstanceOf(PanamaVectorUtilSupport.class);
      assertThat(panama.unavailabilityCause()).isEmpty();
    } else {
      // If SIMD cannot load on this JVM, the cause must be retained for diagnostics.
      assertThat(panama.unavailabilityCause()).isPresent();
    }
  }

  @Test
  void defaultUnavailabilityCauseIsEmpty() {
    VectorUtilSupportProvider provider = FakeProvider.available("x", 1, sentinel());
    assertThat(provider.unavailabilityCause()).isEqualTo(Optional.empty());
  }
}
