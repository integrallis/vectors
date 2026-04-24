package com.integrallis.vectors.server.routing;

import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.ObjectMapperHolder;
import io.helidon.http.HeaderNames;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.util.Map;
import java.util.Objects;

/**
 * Routes for the administrative / observability endpoints: {@code /v1/health}, {@code /v1/metrics}
 * (Prometheus exposition; wired in phase 6).
 *
 * <p>Mounted under the empty prefix by the parent {@link ApiRouting}.
 */
public final class AdminRoutes implements HttpService {

  private final CollectionRegistry registry;

  /**
   * @param registry collection registry used to compute the {@code /v1/health} body
   */
  public AdminRoutes(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  @Override
  public void routing(HttpRules rules) {
    rules.get("/v1/health", this::health);
  }

  private void health(ServerRequest req, ServerResponse res) {
    try {
      String body =
          ObjectMapperHolder.shared()
              .writeValueAsString(Map.of("status", "UP", "collections", registry.size()));
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.send(body);
    } catch (Exception e) {
      res.status(io.helidon.http.Status.INTERNAL_SERVER_ERROR_500).send(e.getMessage());
    }
  }
}
