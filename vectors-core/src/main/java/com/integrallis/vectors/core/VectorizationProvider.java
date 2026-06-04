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
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Selects the fastest available {@link VectorUtilSupport} implementation at class-load time.
 * Prefers the Panama Vector API SIMD implementation; falls back to scalar if unavailable.
 *
 * <p>Selection order:
 *
 * <ol>
 *   <li>Panama Vector API SIMD implementation -- requires JDK 22+ with {@code jdk.incubator.vector}
 *   <li>Scalar fallback
 * </ol>
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
    // Allow forcing scalar mode for testing
    if (isForcedScalar()) {
      LOG.info("vectors-core: Forced scalar mode via -Dvectors.forceScalar=true");
      return new ScalarVectorUtilSupport();
    }

    try {
      VectorUtilSupport panama = new PanamaVectorUtilSupport();
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
      return panama;
    } catch (Throwable t) {
      // Record the throwable (type + stack), not just getMessage() — a removed/renamed
      // jdk.incubator.vector after a JDK upgrade often surfaces with a null/empty message, and the
      // silent scalar fallback is exactly the regression P1.7 guards against. Query via
      // getPanamaFailure(). Fallback behaviour is unchanged.
      panamaFailure = t;
      LOG.log(
          Level.WARNING,
          "vectors-core: Panama Vector API not available, falling back to scalar",
          t);
      return new ScalarVectorUtilSupport();
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
