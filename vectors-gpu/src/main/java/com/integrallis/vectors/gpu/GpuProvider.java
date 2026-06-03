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

import java.util.ServiceLoader;
import java.util.logging.Logger;

/**
 * Runtime entry point for the optional GPU acceleration path. Resolves the first available {@link
 * GpuBackend} via {@link ServiceLoader}; when none is available (no CUDA driver, {@code libcuvs.so}
 * missing, or {@code cuvs-java} excluded from the classpath) callers receive a {@link
 * GpuAvailability#unavailable(String) unavailable} result and should fall back to the CPU indexes.
 *
 * <p>The first successful probe is cached; subsequent calls are O(1).
 */
public final class GpuProvider {

  private static final Logger LOG = Logger.getLogger(GpuProvider.class.getName());

  private static final Resolved RESOLVED = resolve(ServiceLoader.load(GpuBackend.class));

  private GpuProvider() {}

  /** Reports whether any GPU backend initialised successfully. */
  public static GpuAvailability availability() {
    return RESOLVED.availability();
  }

  /**
   * Returns the selected backend, or {@code null} when {@link #availability()} reports unavailable.
   * Callers SHOULD guard with {@link GpuAvailability#isAvailable()}.
   */
  public static GpuBackend backend() {
    return RESOLVED.backend();
  }

  static Resolved resolve(Iterable<GpuBackend> backends) {
    for (GpuBackend candidate : backends) {
      GpuAvailability a = candidate.detect();
      if (a.isAvailable()) {
        LOG.info("vectors-gpu: selected backend " + candidate.name() + " (" + a + ")");
        return new Resolved(a, candidate);
      }
      LOG.fine(
          "vectors-gpu: backend " + candidate.name() + " unavailable: " + a.reason().orElse("?"));
    }
    return new Resolved(
        GpuAvailability.unavailable("no GpuBackend service advertised compatibility"), null);
  }

  record Resolved(GpuAvailability availability, GpuBackend backend) {}
}
