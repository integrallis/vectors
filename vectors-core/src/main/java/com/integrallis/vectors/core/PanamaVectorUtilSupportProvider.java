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
import java.util.logging.Logger;

/**
 * Built-in {@link VectorUtilSupportProvider} for the Panama Vector API ({@code
 * jdk.incubator.vector}) SIMD kernels. Highest-priority of the two built-ins, so SIMD stays the
 * default when available.
 *
 * <p>Availability is probed by actually constructing {@link PanamaVectorUtilSupport} (the only
 * reliable signal that the incubator module is present and the static SIMD constants initialise).
 * The probe instance is discarded; {@link #create()} builds a fresh one, which is cheap because the
 * heavy SIMD constants live in static fields computed once at class-load.
 */
public final class PanamaVectorUtilSupportProvider implements VectorUtilSupportProvider {

  private static final Logger LOG =
      Logger.getLogger(PanamaVectorUtilSupportProvider.class.getName());

  private Throwable failure;

  /** Public no-arg constructor required by {@link java.util.ServiceLoader}. */
  public PanamaVectorUtilSupportProvider() {}

  @Override
  public String name() {
    return "panama";
  }

  @Override
  public int priority() {
    return PANAMA_PRIORITY;
  }

  @Override
  public boolean isAvailable() {
    if (failure != null) {
      return false;
    }
    try {
      // Probe only — a successful construction proves the incubator module loaded.
      VectorUtilSupport probe = new PanamaVectorUtilSupport();
      LOG.fine(
          () ->
              "vectors-core: Panama SIMD probe succeeded ("
                  + probe.getClass().getSimpleName()
                  + ")");
      return true;
    } catch (Throwable t) {
      // A removed/renamed jdk.incubator.vector after a JDK upgrade often surfaces with a null/empty
      // message; keep the throwable (type + stack) so the silent scalar fallback is diagnosable.
      failure = t;
      return false;
    }
  }

  @Override
  public VectorUtilSupport create() {
    VectorUtilSupport instance = new PanamaVectorUtilSupport();
    String capNote =
        PanamaVectorUtilSupport.VECTOR_BITSIZE < PanamaConstants.PREFERRED_BITS
            ? " (capped from hardware-preferred "
                + PanamaConstants.PREFERRED_BITS
                + "; raise with -Dvectors.maxBits)"
            : "";
    LOG.info(
        "vectors-core: Using Panama Vector API SIMD provider (vector bits: "
            + PanamaVectorUtilSupport.VECTOR_BITSIZE
            + capNote
            + ")");
    return instance;
  }

  @Override
  public Optional<Throwable> unavailabilityCause() {
    return Optional.ofNullable(failure);
  }
}
