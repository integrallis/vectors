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

import java.util.Optional;

/**
 * Result of probing for GPU support at runtime. Either reports {@link #isAvailable() available}
 * with a provider name and device count, or unavailable with a human-readable reason.
 *
 * <p>Produced by {@link GpuBackend#detect()} on a best-effort basis; callers use it to decide
 * whether to fall back to the CPU indexes in {@code vectors-hnsw} / {@code vectors-vamana} / {@code
 * vectors-ivf}.
 */
public final class GpuAvailability {

  private final boolean available;
  private final String providerName;
  private final int deviceCount;
  private final String reason;

  private GpuAvailability(boolean available, String providerName, int deviceCount, String reason) {
    this.available = available;
    this.providerName = providerName;
    this.deviceCount = deviceCount;
    this.reason = reason;
  }

  /** Reports that {@code providerName} is usable and exposed {@code deviceCount} CUDA devices. */
  public static GpuAvailability available(String providerName, int deviceCount) {
    return new GpuAvailability(true, providerName, deviceCount, null);
  }

  /** Reports that no GPU backend could be initialised; {@code reason} explains why. */
  public static GpuAvailability unavailable(String reason) {
    return new GpuAvailability(false, null, 0, reason);
  }

  public boolean isAvailable() {
    return available;
  }

  public Optional<String> providerName() {
    return Optional.ofNullable(providerName);
  }

  public int deviceCount() {
    return deviceCount;
  }

  public Optional<String> reason() {
    return Optional.ofNullable(reason);
  }

  @Override
  public String toString() {
    return available
        ? "GpuAvailability{available=true, provider="
            + providerName
            + ", devices="
            + deviceCount
            + "}"
        : "GpuAvailability{available=false, reason=" + reason + "}";
  }
}
