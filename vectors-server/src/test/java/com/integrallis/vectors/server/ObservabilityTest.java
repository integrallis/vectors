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
package com.integrallis.vectors.server;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class ObservabilityTest {

  private static final ObjectMapper JSON = ObjectMapperHolder.shared();

  private VectorsServer.ServerHandle handle;
  private HttpClient client;

  @BeforeEach
  void setUp() throws Exception {
    handle = VectorsServer.start(ServerConfig.forTesting());
    client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    assertThat(
            postJson(
                    "/v1/collections",
                    "{\"name\":\"obs\",\"dimension\":4,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}")
                .statusCode())
        .isEqualTo(201);
  }

  @AfterEach
  void tearDown() {
    if (handle != null) handle.close();
  }

  private URI uri(String path) {
    return URI.create("http://localhost:" + handle.port() + path);
  }

  private HttpResponse<String> postJson(String path, String body) throws Exception {
    return client.send(
        HttpRequest.newBuilder(uri(path))
            .timeout(Duration.ofSeconds(10))
            .header("content-type", "application/json")
            .POST(BodyPublishers.ofString(body))
            .build(),
        BodyHandlers.ofString());
  }

  private HttpResponse<String> get(String path) throws Exception {
    return client.send(
        HttpRequest.newBuilder(uri(path)).timeout(Duration.ofSeconds(5)).GET().build(),
        BodyHandlers.ofString());
  }

  @Test
  void readyzReturnsPlainText() throws Exception {
    HttpResponse<String> r = get("/v1/readyz");
    assertThat(r.statusCode()).isEqualTo(200);
    assertThat(r.body()).contains("READY");
  }

  @Test
  void metricsExposePrometheusFormat() throws Exception {
    HttpResponse<String> r = get("/v1/metrics");
    assertThat(r.statusCode()).isEqualTo(200);
    assertThat(r.headers().firstValue("content-type"))
        .hasValueSatisfying(v -> assertThat(v).startsWith("text/plain"));
    String body = r.body();
    assertThat(body)
        .contains("# HELP vectors_server_uptime_seconds")
        .contains("# TYPE vectors_server_heap_used_bytes gauge")
        .contains("vectors_server_collections_total 1")
        .contains("vectors_collection_size{collection=\"obs\"}")
        .contains("vectors_collection_epoch{collection=\"obs\"} 0");
  }

  @Test
  void epochReturns200WithEtagAnd304OnMatch() throws Exception {
    HttpResponse<String> first = get("/v1/collections/obs/epoch");
    assertThat(first.statusCode()).isEqualTo(200);
    String etag = first.headers().firstValue("etag").orElseThrow();
    assertThat(etag).isEqualTo("\"0\"");
    JsonNode body = JSON.readTree(first.body());
    assertThat(body.get("epoch").asLong()).isZero();
    assertThat(body.get("name").asText()).isEqualTo("obs");

    HttpRequest conditional =
        HttpRequest.newBuilder(uri("/v1/collections/obs/epoch"))
            .timeout(Duration.ofSeconds(5))
            .header("if-none-match", etag)
            .GET()
            .build();
    HttpResponse<String> notMod = client.send(conditional, BodyHandlers.ofString());
    assertThat(notMod.statusCode()).isEqualTo(304);
  }

  @Test
  void epochAdvancesOnCommit() throws Exception {
    assertThat(get("/v1/collections/obs/epoch").body()).contains("\"epoch\":0");

    assertThat(
            postJson(
                    "/v1/collections/obs/documents",
                    "{\"documents\":[{\"id\":\"a\",\"vector\":[1,0,0,0]}]}")
                .statusCode())
        .isEqualTo(200);

    assertThat(get("/v1/collections/obs/epoch").body()).contains("\"epoch\":1");

    assertThat(
            postJson(
                    "/v1/collections/obs/documents",
                    "{\"documents\":[{\"id\":\"b\",\"vector\":[0,1,0,0]}]}")
                .statusCode())
        .isEqualTo(200);
    assertThat(get("/v1/collections/obs/epoch").body()).contains("\"epoch\":2");
  }

  @Test
  void epochUnknownCollectionReturns404() throws Exception {
    HttpResponse<String> r = get("/v1/collections/nope/epoch");
    assertThat(r.statusCode()).isEqualTo(404);
  }

  @Test
  void healthEndpointStillWorks() throws Exception {
    HttpResponse<String> r = get("/v1/health");
    assertThat(r.statusCode()).isEqualTo(200);
    assertThat(r.body()).contains("\"status\":\"UP\"").contains("\"collections\":1");
  }
}
