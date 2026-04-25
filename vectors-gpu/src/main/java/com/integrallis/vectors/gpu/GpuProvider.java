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

import java.util.Iterator;
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

  private static final GpuAvailability AVAILABILITY = probe();
  private static final GpuBackend BACKEND = AVAILABILITY.isAvailable() ? findBackend() : null;

  private GpuProvider() {}

  /** Reports whether any GPU backend initialised successfully. */
  public static GpuAvailability availability() {
    return AVAILABILITY;
  }

  /**
   * Returns the selected backend, or {@code null} when {@link #availability()} reports unavailable.
   * Callers SHOULD guard with {@link GpuAvailability#isAvailable()}.
   */
  public static GpuBackend backend() {
    return BACKEND;
  }

  private static GpuAvailability probe() {
    for (GpuBackend candidate : ServiceLoader.load(GpuBackend.class)) {
      GpuAvailability a = candidate.detect();
      if (a.isAvailable()) {
        LOG.info("vectors-gpu: selected backend " + candidate.name() + " (" + a + ")");
        return a;
      }
      LOG.fine(
          "vectors-gpu: backend " + candidate.name() + " unavailable: " + a.reason().orElse("?"));
    }
    return GpuAvailability.unavailable("no GpuBackend service advertised compatibility");
  }

  private static GpuBackend findBackend() {
    Iterator<GpuBackend> it = ServiceLoader.load(GpuBackend.class).iterator();
    while (it.hasNext()) {
      GpuBackend b = it.next();
      if (b.detect().isAvailable()) {
        return b;
      }
    }
    return null;
  }
}
