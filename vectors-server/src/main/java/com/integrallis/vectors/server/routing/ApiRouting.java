package com.integrallis.vectors.server.routing;

import com.integrallis.vectors.server.CollectionRegistry;
import io.helidon.webserver.http.HttpRouting;
import java.util.Objects;

/**
 * Assembles the complete HTTP routing tree for the server.
 *
 * <p>Phase 1 wires only {@link AdminRoutes}. Subsequent phases attach {@code CollectionsRoutes},
 * {@code DocumentsRoutes}, and {@code SearchRoutes} here.
 */
public final class ApiRouting {

  private final CollectionRegistry registry;

  /**
   * @param registry the collection registry backing the API
   */
  public ApiRouting(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  /**
   * @param builder the Helidon routing builder to augment
   */
  public void apply(HttpRouting.Builder builder) {
    builder.register(new AdminRoutes(registry));
  }
}
