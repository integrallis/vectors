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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Selects the fastest available {@link VectorUtilSupport} implementation at class-load time.
 *
 * <p>Providers are discovered through {@link ServiceLoader} as {@link VectorUtilSupportProvider}
 * services and ranked by {@link VectorUtilSupportProvider#priority()}; the highest-priority
 * {@linkplain VectorUtilSupportProvider#isAvailable() available} provider wins. The two built-in
 * providers keep the historical default order — Panama Vector API SIMD ({@value
 * VectorUtilSupportProvider#PANAMA_PRIORITY}) ahead of the always-available scalar fallback
 * ({@value VectorUtilSupportProvider#SCALAR_PRIORITY}) — while a future native-AVX, SVE, or GPU
 * provider can register a higher priority to take precedence. If no provider is usable (or the
 * service file is missing), the built-in scalar kernels are used so the library never fails to
 * load.
 *
 * <p>{@code -Dvectors.forceScalar=true} bypasses discovery entirely and pins the scalar kernels.
 */
public final class VectorizationProvider {

  private static final Logger LOG = Logger.getLogger(VectorizationProvider.class.getName());

  /**
   * The throwable that prevented the Panama provider from loading, or {@code null} when SIMD is
   * active (or scalar was requested explicitly). Written once inside {@link #selectProvider()}
   * during {@link #INSTANCE} class-initialization — a single writer with a happens-before edge to
   * every subsequent read, so {@code volatile} suffices.
   */
  private static volatile Throwable panamaFailure;

  private static final VectorUtilSupport INSTANCE = selectProvider();

  private VectorizationProvider() {}

  /** Returns the singleton VectorUtilSupport implementation. Thread-safe after class loading. */
  public static VectorUtilSupport getInstance() {
    return INSTANCE;
  }

  private static VectorUtilSupport selectProvider() {
    // Allow forcing scalar mode for testing — bypasses discovery entirely.
    if (isForcedScalar()) {
      LOG.info("vectors-core: Forced scalar mode via -Dvectors.forceScalar=true");
      return new ScalarVectorUtilSupport();
    }
    List<VectorUtilSupportProvider> providers = new ArrayList<>();
    ServiceLoader.load(
            VectorUtilSupportProvider.class, VectorizationProvider.class.getClassLoader())
        .forEach(providers::add);
    return select(providers);
  }

  /**
   * Ranks the given providers by descending {@link VectorUtilSupportProvider#priority()} and
   * returns the kernels of the first one that is available and constructs successfully, falling
   * back to the built-in scalar kernels if none qualifies. Package-private and pure (aside from
   * recording a Panama load failure for diagnostics) so the selection policy is unit-testable with
   * synthetic providers.
   */
  static VectorUtilSupport select(List<VectorUtilSupportProvider> providers) {
    List<VectorUtilSupportProvider> ranked = new ArrayList<>(providers);
    // Stable sort keeps service-discovery order as the tie-breaker for equal priorities.
    ranked.sort(Comparator.comparingInt(VectorUtilSupportProvider::priority).reversed());
    for (VectorUtilSupportProvider provider : ranked) {
      boolean available;
      try {
        available = provider.isAvailable();
      } catch (Throwable t) {
        recordPanamaFailure(provider, t);
        LOG.log(
            Level.WARNING,
            "vectors-core: provider '" + safeName(provider) + "' threw during availability probe",
            t);
        continue;
      }
      if (!available) {
        Optional<Throwable> cause = provider.unavailabilityCause();
        if (cause.isPresent()) {
          recordPanamaFailure(provider, cause.get());
          LOG.log(
              Level.WARNING,
              "vectors-core: provider '" + safeName(provider) + "' unavailable, trying next",
              cause.get());
        } else {
          LOG.fine(
              "vectors-core: provider '" + safeName(provider) + "' not available, trying next");
        }
        continue;
      }
      try {
        VectorUtilSupport impl = provider.create();
        if (impl == null) {
          LOG.warning(
              "vectors-core: provider '" + safeName(provider) + "' returned null, trying next");
          continue;
        }
        return impl;
      } catch (Throwable t) {
        recordPanamaFailure(provider, t);
        LOG.log(
            Level.WARNING,
            "vectors-core: provider '" + safeName(provider) + "' failed to create, trying next",
            t);
      }
    }
    LOG.warning("vectors-core: no VectorUtilSupportProvider available; using scalar fallback");
    return new ScalarVectorUtilSupport();
  }

  /**
   * Records a Panama load failure so {@link #getPanamaFailure()} can surface a silent
   * SIMD-to-scalar regression (P1.7). Only the built-in Panama provider feeds this diagnostic.
   */
  private static void recordPanamaFailure(VectorUtilSupportProvider provider, Throwable t) {
    if (provider instanceof PanamaVectorUtilSupportProvider) {
      panamaFailure = t;
    }
  }

  private static String safeName(VectorUtilSupportProvider provider) {
    try {
      return provider.name();
    } catch (Throwable t) {
      return provider.getClass().getSimpleName();
    }
  }

  /**
   * Returns the throwable that prevented the Panama SIMD provider from loading, present only when
   * the runtime silently fell back to the scalar implementation. Empty when SIMD is active or when
   * scalar was requested explicitly via {@code -Dvectors.forceScalar}. Lets callers detect a SIMD
   * regression (e.g. a JDK upgrade that drops {@code jdk.incubator.vector}) instead of paying an
   * unexplained throughput cliff.
   */
  public static Optional<Throwable> getPanamaFailure() {
    return Optional.ofNullable(panamaFailure);
  }

  /** Returns true when scalar mode was requested explicitly via {@code -Dvectors.forceScalar}. */
  public static boolean isForcedScalar() {
    return Boolean.getBoolean("vectors.forceScalar");
  }

  /** Returns true if the current provider is the SIMD (Panama) implementation. */
  public static boolean isPanamaEnabled() {
    return INSTANCE instanceof PanamaVectorUtilSupport;
  }

  /** Returns the name of the current provider for diagnostics. */
  public static String getProviderName() {
    return INSTANCE.getClass().getSimpleName();
  }

  /**
   * Creates a scalar-only provider instance for testing or comparison purposes. Not cached -- each
   * call returns a new instance.
   */
  public static VectorUtilSupport newScalarProvider() {
    return new ScalarVectorUtilSupport();
  }
}
