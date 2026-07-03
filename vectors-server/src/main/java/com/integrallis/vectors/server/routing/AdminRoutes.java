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

import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.ObjectMapperHolder;
import io.helidon.http.HeaderNames;
import io.helidon.http.Status;
import io.helidon.webserver.http.HttpRules;
import io.helidon.webserver.http.HttpService;
import io.helidon.webserver.http.ServerRequest;
import io.helidon.webserver.http.ServerResponse;
import java.lang.management.ManagementFactory;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Administrative and observability endpoints:
 *
 * <ul>
 *   <li>{@code GET /v1/health} — liveness probe, {@code {"status":"UP","collections":N}}.
 *   <li>{@code GET /v1/readyz} — readiness probe; {@code 200} once startup discovery completed.
 *   <li>{@code GET /v1/metrics} — Prometheus text-exposition format (version 0.0.4).
 *   <li>{@code GET /v1/collections/\u007Bname\u007D/epoch} — ETag-cacheable integer epoch that
 *       advances on every successful commit. Responds {@code 304 Not Modified} when the client's
 *       {@code If-None-Match} matches.
 * </ul>
 */
public final class AdminRoutes implements HttpService {

  private static final Logger LOG = LoggerFactory.getLogger(AdminRoutes.class);

  private final CollectionRegistry registry;

  public AdminRoutes(CollectionRegistry registry) {
    this.registry = Objects.requireNonNull(registry, "registry");
  }

  @Override
  public void routing(HttpRules rules) {
    rules
        .get("/v1/health", this::health)
        .get("/v1/readyz", this::readyz)
        .get("/v1/metrics", this::metrics)
        .get("/v1/collections/{name}/epoch", this::epoch);
  }

  private void health(ServerRequest req, ServerResponse res) {
    try {
      String body =
          ObjectMapperHolder.shared()
              .writeValueAsString(Map.of("status", "UP", "collections", registry.size()));
      res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
      res.send(body);
    } catch (Exception e) {
      LOG.error("failed to serialize health payload", e);
      RouteSupport.sendProblem(
          res, Status.INTERNAL_SERVER_ERROR_500, "internal error", "see server logs", req);
    }
  }

  private void readyz(ServerRequest req, ServerResponse res) {
    // With synchronous startup discovery, the server is always ready by the time requests arrive.
    res.headers().set(HeaderNames.CONTENT_TYPE, "text/plain; charset=utf-8");
    res.send("READY\n");
  }

  private void epoch(ServerRequest req, ServerResponse res) {
    String name = req.path().pathParameters().get("name");
    if (!RouteSupport.validateName(name, req, res)) return;
    Optional<VectorCollection> col = registry.get(name);
    if (col.isEmpty()) {
      RouteSupport.sendProblem(res, Status.NOT_FOUND_404, "collection not found", name, req);
      return;
    }
    long epoch = registry.epoch(name);
    String etag = "\"" + epoch + "\"";
    String incoming = req.headers().first(HeaderNames.IF_NONE_MATCH).orElse(null);
    if (incoming != null && incoming.equals(etag)) {
      res.headers().set(HeaderNames.ETAG, etag);
      res.status(Status.NOT_MODIFIED_304).send();
      return;
    }
    res.headers().set(HeaderNames.ETAG, etag);
    res.headers().set(HeaderNames.CONTENT_TYPE, "application/json");
    try {
      res.send(
          ObjectMapperHolder.shared().writeValueAsString(Map.of("name", name, "epoch", epoch)));
    } catch (Exception e) {
      LOG.error("failed to serialize epoch payload for '{}'", name, e);
      RouteSupport.sendProblem(
          res, Status.INTERNAL_SERVER_ERROR_500, "internal error", "see server logs", req);
    }
  }

  private void metrics(ServerRequest req, ServerResponse res) {
    StringBuilder sb = new StringBuilder(1024);
    // Process / JVM basics — no guava, no micrometer; hand-rolled to keep dependencies minimal.
    long uptimeMs = ManagementFactory.getRuntimeMXBean().getUptime();
    Runtime rt = Runtime.getRuntime();
    long heapUsed = rt.totalMemory() - rt.freeMemory();
    long heapMax = rt.maxMemory();
    int threads = Thread.activeCount();
    int collections = registry.size();

    appendGauge(sb, "vectors_server_uptime_seconds", "Server uptime in seconds", uptimeMs / 1000.0);
    appendGauge(sb, "vectors_server_heap_used_bytes", "JVM used heap bytes", heapUsed);
    appendGauge(sb, "vectors_server_heap_max_bytes", "JVM maximum heap bytes", heapMax);
    appendGauge(sb, "vectors_server_threads", "Current active thread count", threads);
    appendGauge(
        sb,
        "vectors_server_collections_total",
        "Number of currently open collections",
        collections);

    // Per-collection labelled metrics: size and epoch.
    sb.append("# HELP vectors_collection_size Number of live documents in a collection\n");
    sb.append("# TYPE vectors_collection_size gauge\n");
    for (String name : registry.names()) {
      registry
          .get(name)
          .ifPresent(
              c ->
                  sb.append("vectors_collection_size{collection=\"")
                      .append(escape(name))
                      .append("\"} ")
                      .append(c.size())
                      .append('\n'));
    }
    sb.append("# HELP vectors_collection_epoch Monotonic commit-epoch counter\n");
    sb.append("# TYPE vectors_collection_epoch counter\n");
    for (String name : registry.names()) {
      sb.append("vectors_collection_epoch{collection=\"")
          .append(escape(name))
          .append("\"} ")
          .append(registry.epoch(name))
          .append('\n');
    }

    res.headers().set(HeaderNames.CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8");
    res.send(sb.toString());
  }

  private static void appendGauge(StringBuilder sb, String name, String help, double value) {
    sb.append("# HELP ").append(name).append(' ').append(help).append('\n');
    sb.append("# TYPE ").append(name).append(" gauge\n");
    sb.append(name).append(' ').append(formatValue(value)).append('\n');
  }

  private static String formatValue(double v) {
    if (v == Math.floor(v) && !Double.isInfinite(v)) {
      return Long.toString((long) v);
    }
    return Double.toString(v);
  }

  /**
   * Escapes a string for use inside a Prometheus label value (double-quoted). Collection names are
   * validated against {@code [A-Za-z0-9_-]{1,128}} so in practice this is a no-op, but we escape
   * defensively in case the validation is ever relaxed.
   */
  private static String escape(String s) {
    return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
  }
}
