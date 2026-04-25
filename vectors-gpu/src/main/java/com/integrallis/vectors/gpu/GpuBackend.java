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

/**
 * SPI for a GPU-accelerated vector search backend. Implementations wrap a native library (e.g.
 * NVIDIA cuVS) and are loaded via {@link java.util.ServiceLoader}.
 *
 * <p>Discovery order (first {@linkplain GpuAvailability#isAvailable() available} wins):
 *
 * <ol>
 *   <li>{@link CuVsGpuBackend} -- Panama-FFM bindings to {@code libcuvs.so}
 * </ol>
 *
 * <p>Implementations MUST NOT throw from {@link #detect()}: any failure to locate or initialise the
 * native runtime is reported as {@link GpuAvailability#unavailable(String)}.
 */
public interface GpuBackend {

  /** Stable identifier for the backend, e.g. {@code "cuvs"}. */
  String name();

  /** Probes the host for GPU support. MUST be side-effect-free beyond loading native libraries. */
  GpuAvailability detect();
}
