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
package com.integrallis.vectors.studio.web;

import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.sidecart.SidecartRegistry;
import com.integrallis.vectors.studio.web.embed.ProviderRegistry;

/**
 * Per-instance configuration for the Studio web server. {@code sidecart} may be empty; when set, it
 * lets the web layer resolve image / text / binary payloads from external (sidecart) data sources
 * keyed by document id, layered on top of whatever the active {@link StudioSession} backend can
 * already serve. {@code providerRegistry} may be {@code null}, in which case the server loads the
 * bundled {@code embedding-providers.json} (plus any {@code VECTORS_STUDIO_PROVIDERS} override);
 * tests may inject a custom registry as a seam for query-time embedding.
 */
public record StudioConfig(
    int port, StudioSession session, SidecartRegistry sidecart, ProviderRegistry providerRegistry) {

  /** Default listen port (8288, mirroring the {@code vectors-server} convention). */
  public static final int DEFAULT_PORT = 8288;

  /** Convenience constructor — no sidecart bindings, default provider registry. */
  public StudioConfig(int port, StudioSession session) {
    this(port, session, SidecartRegistry.empty(), null);
  }

  /** Convenience constructor — sidecart bindings, default provider registry. */
  public StudioConfig(int port, StudioSession session, SidecartRegistry sidecart) {
    this(port, session, sidecart, null);
  }
}
