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

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * GPU backend backed by NVIDIA cuVS Panama-FFM bindings. Detection attempts to resolve {@code
 * com.nvidia.cuvs.spi.CuVSProvider} reflectively so that the module builds and loads on hosts
 * without {@code libcuvs.so} (detection simply reports unavailable).
 *
 * <p>Search and indexing entry points are intentionally absent in this scaffold; they are added
 * incrementally once the CPU-side surface (Phase 3 of the roadmap) stabilises.
 */
public final class CuVsGpuBackend implements GpuBackend {

  private static final Logger LOG = Logger.getLogger(CuVsGpuBackend.class.getName());

  private static final String PROVIDER_CLASS = "com.nvidia.cuvs.spi.CuVSProvider";

  @Override
  public String name() {
    return "cuvs";
  }

  @Override
  public GpuAvailability detect() {
    if (Boolean.getBoolean("vectors.gpu.disable")) {
      return GpuAvailability.unavailable("disabled via -Dvectors.gpu.disable=true");
    }

    Class<?> providerClass;
    try {
      providerClass = Class.forName(PROVIDER_CLASS);
    } catch (ClassNotFoundException e) {
      return GpuAvailability.unavailable(
          "cuvs-java not on classpath (" + PROVIDER_CLASS + " not found)");
    }

    try {
      Object provider = providerClass.getMethod("provider").invoke(null);
      Object infoProvider = providerClass.getMethod("gpuInfoProvider").invoke(provider);
      Object compatible = infoProvider.getClass().getMethod("compatibleGPUs").invoke(infoProvider);
      int count = ((java.util.List<?>) compatible).size();
      if (count == 0) {
        return GpuAvailability.unavailable("cuvs loaded but no compatible CUDA devices found");
      }
      return GpuAvailability.available(name(), count);
    } catch (Throwable t) {
      // ExceptionInInitializerError, UnsatisfiedLinkError, or the provider's own init failure
      // when libcuvs.so is missing. All route through the same unavailable path.
      Throwable cause = t.getCause() != null ? t.getCause() : t;
      LOG.log(Level.FINE, "cuVS GPU detection failed", cause);
      return GpuAvailability.unavailable(
          cause.getClass().getSimpleName() + ": " + cause.getMessage());
    }
  }
}
