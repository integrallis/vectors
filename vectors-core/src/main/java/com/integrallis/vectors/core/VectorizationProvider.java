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

import java.util.logging.Logger;

/**
 * Selects the fastest available {@link VectorUtilSupport} implementation at class-load time.
 * Prefers the Panama Vector API SIMD implementation; falls back to scalar if unavailable.
 *
 * <p>Selection order:
 *
 * <ol>
 *   <li>{@link PanamaVectorUtilSupport} -- requires JDK 22+ with {@code jdk.incubator.vector}
 *   <li>{@link ScalarVectorUtilSupport} -- pure scalar fallback
 * </ol>
 */
public final class VectorizationProvider {

  private static final Logger LOG = Logger.getLogger(VectorizationProvider.class.getName());

  private static final VectorUtilSupport INSTANCE = selectProvider();

  private VectorizationProvider() {}

  /** Returns the singleton VectorUtilSupport implementation. Thread-safe after class loading. */
  public static VectorUtilSupport getInstance() {
    return INSTANCE;
  }

  private static VectorUtilSupport selectProvider() {
    // Allow forcing scalar mode for testing
    if (Boolean.getBoolean("vectors.forceScalar")) {
      LOG.info("vectors-core: Forced scalar mode via -Dvectors.forceScalar=true");
      return new ScalarVectorUtilSupport();
    }

    try {
      VectorUtilSupport panama = new PanamaVectorUtilSupport();
      LOG.info(
          "vectors-core: Using Panama Vector API SIMD provider (bit size: "
              + PanamaVectorUtilSupport.VECTOR_BITSIZE
              + ")");
      return panama;
    } catch (Throwable t) {
      LOG.warning(
          "vectors-core: Panama Vector API not available, falling back to scalar: "
              + t.getMessage());
      return new ScalarVectorUtilSupport();
    }
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
