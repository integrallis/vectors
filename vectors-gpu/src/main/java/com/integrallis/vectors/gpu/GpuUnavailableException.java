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
 * Thrown when a GPU-backed index is constructed or used on a host where {@link GpuProvider} reports
 * {@link GpuAvailability#isAvailable()} false. Wraps the best-effort reason string reported by the
 * backend (e.g. {@code "cuvs-java not on classpath"}, {@code "UnsatisfiedLinkError: libcuvs.so"},
 * {@code "no compatible CUDA devices found"}).
 */
public final class GpuUnavailableException extends RuntimeException {

  private static final long serialVersionUID = 1L;

  public GpuUnavailableException(String reason) {
    super("GPU backend not available: " + reason);
  }

  public GpuUnavailableException(String reason, Throwable cause) {
    super("GPU backend not available: " + reason, cause);
  }
}
