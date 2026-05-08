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
package com.integrallis.vectors.studio.core.connection;

import java.util.Optional;

/**
 * SPI for opening additional {@link StudioBackend} implementations from a {@link ConnectionConfig}.
 * Providers are discovered by {@link StudioBackendFactory} via {@link java.util.ServiceLoader} so
 * downstream modules (e.g. distributed / R2-backed backends) can register themselves without a
 * compile-time dependency in {@code vectors-studio-core}.
 */
public interface StudioBackendProvider {

  /**
   * Returns a non-empty optional when this provider can open {@code cfg}, or {@link
   * Optional#empty()} otherwise. Implementations should return empty (not throw) when the
   * configuration is for a different provider.
   */
  Optional<StudioBackend> tryOpen(ConnectionConfig cfg);
}
