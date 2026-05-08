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
package com.integrallis.vectors.studio.distributed;

import com.integrallis.vectors.studio.core.connection.ConnectionConfig;
import com.integrallis.vectors.studio.core.connection.StudioBackend;
import com.integrallis.vectors.studio.core.connection.StudioBackendProvider;
import java.util.Optional;

/**
 * {@link StudioBackendProvider} SPI implementation: opens a {@link DistributedStudioBackend}
 * whenever the supplied {@link ConnectionConfig} is a {@link DistributedConnectionConfig}.
 * Discovered by {@link com.integrallis.vectors.studio.core.connection.StudioBackendFactory} via
 * {@link java.util.ServiceLoader}.
 */
public final class DistributedStudioBackendProvider implements StudioBackendProvider {

  @Override
  public Optional<StudioBackend> tryOpen(ConnectionConfig cfg) {
    if (cfg instanceof DistributedConnectionConfig d) {
      return Optional.of(DistributedStudioBackend.open(d));
    }
    return Optional.empty();
  }
}
