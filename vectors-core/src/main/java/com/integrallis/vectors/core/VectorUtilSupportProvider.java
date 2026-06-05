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

import java.util.Optional;

/**
 * Service-provider interface for {@link VectorUtilSupport} kernel implementations.
 *
 * <p>Providers are discovered at class-load time via {@link java.util.ServiceLoader} (declared in
 * {@code META-INF/services/com.integrallis.vectors.core.VectorUtilSupportProvider}) and ranked by
 * {@link #priority()} — the highest-priority {@linkplain #isAvailable() available} provider wins.
 * {@link VectorizationProvider} probes them in order and falls back to the built-in scalar kernels
 * if none is usable, so a missing or empty service file never breaks the library.
 *
 * <p>The two built-in providers establish the default ranking:
 *
 * <ul>
 *   <li>{@link #PANAMA_PRIORITY} (100) — the Panama Vector API SIMD kernels, available on a JDK
 *       with {@code jdk.incubator.vector}. This keeps SIMD the default, exactly as before this SPI
 *       existed.
 *   <li>{@link #SCALAR_PRIORITY} (0) — the pure-Java scalar kernels, always available.
 * </ul>
 *
 * <p>A future native-AVX, SVE, or GPU provider plugs in by shipping its own jar with a service
 * entry and a {@code priority()} above {@link #PANAMA_PRIORITY} to take precedence, or between the
 * two built-ins to slot in only where SIMD is unavailable. Implementations must have a public
 * no-arg constructor.
 */
public interface VectorUtilSupportProvider {

  /** Priority of the built-in Panama (SIMD) provider; the default selection when available. */
  int PANAMA_PRIORITY = 100;

  /** Priority of the built-in scalar provider; the always-available fallback. */
  int SCALAR_PRIORITY = 0;

  /** A short, stable identifier for diagnostics and logging (e.g. {@code "panama"}). */
  String name();

  /**
   * Selection rank; higher wins. See {@link #PANAMA_PRIORITY} and {@link #SCALAR_PRIORITY} for the
   * built-in anchors. Ties are broken by service-discovery order.
   */
  int priority();

  /**
   * Probes whether this provider's kernels can run on the current JVM and hardware. Must not throw
   * — a provider that cannot determine availability should return {@code false} and may expose the
   * reason via {@link #unavailabilityCause()}. A provider that returns {@code true} must be able to
   * satisfy a subsequent {@link #create()} call.
   */
  boolean isAvailable();

  /**
   * Instantiates the kernel implementation. Only called after {@link #isAvailable()} returns {@code
   * true}; may still throw, in which case {@link VectorizationProvider} skips to the next provider.
   */
  VectorUtilSupport create();

  /**
   * The throwable that made this provider unavailable, if any. Lets {@link VectorizationProvider}
   * surface a silent SIMD-to-scalar regression (e.g. a JDK upgrade that drops {@code
   * jdk.incubator.vector}) instead of paying an unexplained throughput cliff. Defaults to empty.
   */
  default Optional<Throwable> unavailabilityCause() {
    return Optional.empty();
  }
}
