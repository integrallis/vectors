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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.server.CollectionRegistry;
import com.integrallis.vectors.server.ServerConfig;
import io.helidon.webserver.WebServer;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** An embedding process adds its own gauges to the engine's {@code /v1/metrics}; a real server. */
class MetricsContributorTest {

  private WebServer server;
  private CollectionRegistry registry;

  @AfterEach
  void stop() {
    if (server != null) {
      server.stop();
    }
    if (registry != null) {
      registry.close();
    }
  }

  private String scrape(List<MetricsContributor> contributors) throws Exception {
    registry = new CollectionRegistry();
    ServerConfig config = ServerConfig.forTesting();
    server =
        WebServer.builder()
            .port(0)
            .routing(b -> new ApiRouting(registry, config, contributors).apply(b))
            .build()
            .start();
    HttpResponse<String> r =
        HttpClient.newHttpClient()
            .send(
                HttpRequest.newBuilder(
                        URI.create("http://127.0.0.1:" + server.port() + "/v1/metrics"))
                    .GET()
                    .build(),
                HttpResponse.BodyHandlers.ofString());
    assertThat(r.statusCode()).isEqualTo(200);
    assertThat(r.headers().firstValue("content-type").orElse(""))
        .startsWith("text/plain; version=0.0.4");
    return r.body();
  }

  @Test
  void contributedLinesFollowTheEngineMetrics() throws Exception {
    String body =
        scrape(
            List.of(
                out ->
                    out.append("# HELP embedder_up Whether the embedding process is up\n")
                        .append("# TYPE embedder_up gauge\n")
                        .append("embedder_up 1\n"),
                out -> out.append("embedder_epoch{group=\"cmg\"} 7\n")));
    assertThat(body).contains("vectors_server_uptime_seconds").contains("embedder_up 1\n");
    assertThat(body.indexOf("vectors_server_collections_total"))
        .isLessThan(body.indexOf("embedder_up 1"));
    assertThat(body).contains("embedder_epoch{group=\"cmg\"} 7\n");
  }

  @Test
  void aThrowingContributorIsSkippedAndTheEngineMetricsStillServe() throws Exception {
    String body =
        scrape(
            List.of(
                out -> {
                  throw new IllegalStateException("not ready");
                },
                out -> out.append("after_the_broken_one 1\n")));
    assertThat(body).contains("vectors_server_uptime_seconds").contains("after_the_broken_one 1\n");
    assertThat(body).doesNotContain("not ready");
  }

  @Test
  void theTwoArgumentRoutingIsUnchanged() throws Exception {
    String body = scrape(List.of());
    assertThat(body).contains("vectors_server_uptime_seconds").endsWith("\n");
  }
}
