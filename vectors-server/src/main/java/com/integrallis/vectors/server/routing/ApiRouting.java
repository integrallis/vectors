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
package com.integrallis.vectors.server.routing;

import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.ServerConfig;
import io.helidon.webserver.http.HttpRouting;
import java.util.Objects;

/**
 * Assembles the complete HTTP routing tree for the server.
 *
 * <p>Phase 4 wires {@link AdminRoutes}, {@link CollectionsRoutes}, {@link DocumentsRoutes}, and
 * {@link SearchRoutes}.
 */
public final class ApiRouting {

  private final CollectionRegistry registry;
  private final ServerConfig config;

  /**
   * @param registry the collection registry backing the API
   * @param config server configuration (used by collection-creation routes)
   */
  public ApiRouting(CollectionRegistry registry, ServerConfig config) {
    this.registry = Objects.requireNonNull(registry, "registry");
    this.config = Objects.requireNonNull(config, "config");
  }

  /**
   * @param builder the Helidon routing builder to augment
   */
  public void apply(HttpRouting.Builder builder) {
    builder.register(new AdminRoutes(registry));
    builder.register(new CollectionsRoutes(registry, config));
    builder.register(new DocumentsRoutes(registry));
    builder.register(new SearchRoutes(registry));
    builder.register(new EventsRoutes(registry));
    builder.register(new BulkRoutes(registry));
  }
}
