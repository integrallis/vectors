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
import java.util.ServiceLoader;

/** Single entry point for opening a {@link StudioBackend} from a {@link ConnectionConfig}. */
public final class StudioBackendFactory {

  private StudioBackendFactory() {}

  /**
   * Opens the appropriate backend for the supplied configuration. The built-in {@link
   * ConnectionConfig.Embedded} and {@link ConnectionConfig.Remote} cases are handled directly; any
   * other type is dispatched to a {@link StudioBackendProvider} discovered via {@link
   * ServiceLoader}.
   */
  public static StudioBackend open(ConnectionConfig cfg) {
    if (cfg instanceof ConnectionConfig.Embedded e) return EmbeddedStudioBackend.open(e.dataDir());
    if (cfg instanceof ConnectionConfig.Remote r) return RemoteStudioBackend.open(r);
    for (StudioBackendProvider p : ServiceLoader.load(StudioBackendProvider.class)) {
      Optional<StudioBackend> opt = p.tryOpen(cfg);
      if (opt.isPresent()) return opt.get();
    }
    throw new IllegalArgumentException("no StudioBackendProvider for " + cfg.getClass().getName());
  }
}
