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
package com.integrallis.vectors.server.client;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class VectorsServerClientTest {

  private HttpServer server;
  private ExecutorService executor;
  private VectorsServerClient client;

  @BeforeEach
  void startServer() throws IOException {
    server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    executor = Executors.newCachedThreadPool();
    server.setExecutor(executor);
    client = new VectorsServerClient(baseUrl(), Duration.ofSeconds(2), 1024);
  }

  @AfterEach
  void stopServer() {
    if (client != null) {
      client.close();
    }
    if (server != null) {
      server.stop(0);
    }
    if (executor != null) {
      executor.shutdownNow();
    }
  }

  @Test
  void listCollectionsParsesHappyPath() {
    route(
        "/v1/collections",
        exchange ->
            respond(
                exchange,
                200,
                """
                {"collections":[{"name":"docs","dimension":3,"metric":"COSINE","indexType":"FLAT","quantizer":"NONE","size":2}]}
                """));
    server.start();

    List<CollectionInfo> collections = client.listCollections();

    assertThat(collections).hasSize(1);
    assertThat(collections.getFirst().name()).isEqualTo("docs");
    assertThat(collections.getFirst().dimension()).isEqualTo(3);
    assertThat(collections.getFirst().size()).isEqualTo(2);
  }

  @Test
  void createCollectionSendsJsonAndParsesResponse() {
    AtomicReference<String> requestBody = new AtomicReference<>();
    route(
        "/v1/collections",
        exchange -> {
          requestBody.set(readBody(exchange));
          assertThat(exchange.getRequestMethod()).isEqualTo("POST");
          assertThat(exchange.getRequestHeaders().getFirst("Content-Type"))
              .isEqualTo("application/json");
          respond(
              exchange,
              200,
              """
              {"name":"docs","dimension":3,"metric":"COSINE","indexType":"FLAT","quantizer":"NONE","size":0}
              """);
        });
    server.start();

    CollectionInfo created =
        client.createCollection("docs", 3, "COSINE", "FLAT", Map.of("quantizer", "NONE"));

    assertThat(created.name()).isEqualTo("docs");
    assertThat(requestBody.get()).contains("\"name\":\"docs\"");
    assertThat(requestBody.get()).contains("\"dimension\":3");
    assertThat(requestBody.get()).contains("\"quantizer\":\"NONE\"");
  }

  @Test
  void authenticatedClientSendsBearerToken() {
    AtomicReference<String> authorization = new AtomicReference<>();
    route(
        "/v1/collections",
        exchange -> {
          authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
          respond(
              exchange,
              200,
              """
              {"collections":[]}
              """);
        });
    server.start();
    client.close();
    client = new VectorsServerClient(baseUrl(), Duration.ofSeconds(2), "secret");

    client.listCollections();

    assertThat(authorization.get()).isEqualTo("Bearer secret");
  }

  @Test
  void unauthenticatedClientDoesNotSendAuthorizationHeader() {
    AtomicReference<String> authorization = new AtomicReference<>();
    route(
        "/v1/collections",
        exchange -> {
          authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
          respond(
              exchange,
              200,
              """
              {"collections":[]}
              """);
        });
    server.start();

    client.listCollections();

    assertThat(authorization.get()).isNull();
  }

  @Test
  void blankApiKeyIsRejected() {
    assertThatThrownBy(() -> new VectorsServerClient(baseUrl(), Duration.ofSeconds(2), " "))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("apiKey must not be blank");
  }

  @Test
  void collectionExistsMaps404ToFalse() {
    route("/v1/collections/missing", exchange -> respond(exchange, 404, "{\"error\":\"missing\"}"));
    server.start();

    assertThat(client.collectionExists("missing")).isFalse();
  }

  @Test
  void serverErrorsPreserveStatusAndBody() {
    route("/v1/collections/bad", exchange -> respond(exchange, 500, "{\"error\":\"boom\"}"));
    server.start();

    assertThatThrownBy(() -> client.describe("bad"))
        .isInstanceOf(VectorsServerException.class)
        .satisfies(
            error -> {
              VectorsServerException serverError = (VectorsServerException) error;
              assertThat(serverError.statusCode()).isEqualTo(500);
              assertThat(serverError.body()).isEqualTo("{\"error\":\"boom\"}");
            });
  }

  @Test
  void connectionFailureIsWrapped() {
    server.start();
    int port = server.getAddress().getPort();
    server.stop(0);
    server = null;
    executor.shutdownNow();
    executor = null;
    client = new VectorsServerClient("http://127.0.0.1:" + port, Duration.ofMillis(250), 1024);

    assertThatThrownBy(() -> client.describe("docs"))
        .isInstanceOf(RuntimeException.class)
        .hasMessageContaining("HTTP GET failed");
  }

  @Test
  void requestTimeoutIsAppliedPerRequest() {
    route(
        "/v1/health",
        exchange -> {
          try {
            Thread.sleep(1_000);
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
          }
          respond(exchange, 200, "{}");
        });
    server.start();
    client.close();
    client = new VectorsServerClient(baseUrl(), Duration.ofMillis(100), 1024);

    assertThatThrownBy(() -> client.isHealthy())
        .isInstanceOf(RuntimeException.class)
        .hasMessageContaining("HTTP GET failed");
  }

  @Test
  void responseBodyIsBounded() {
    route("/v1/collections", exchange -> respond(exchange, 200, "0123456789"));
    server.start();
    client.close();
    client = new VectorsServerClient(baseUrl(), Duration.ofSeconds(2), 5);

    assertThatThrownBy(() -> client.listCollections())
        .isInstanceOf(RuntimeException.class)
        .hasMessageContaining("HTTP GET failed");
  }

  private String baseUrl() {
    return "http://127.0.0.1:" + server.getAddress().getPort();
  }

  private void route(String path, HttpHandler handler) {
    server.createContext(path, handler);
  }

  private static String readBody(HttpExchange exchange) throws IOException {
    return new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
  }

  private static void respond(HttpExchange exchange, int status, String body) throws IOException {
    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
    exchange.getResponseHeaders().set("Content-Type", "application/json");
    exchange.sendResponseHeaders(status, bytes.length);
    try (var out = exchange.getResponseBody()) {
      out.write(bytes);
    }
  }
}
