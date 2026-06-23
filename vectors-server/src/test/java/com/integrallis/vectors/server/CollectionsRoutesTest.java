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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.time.Duration;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

class CollectionsRoutesTest {

  private static final ObjectMapper JSON = ObjectMapperHolder.shared();

  private VectorsServer.ServerHandle handle;
  private HttpClient client;

  @BeforeEach
  void startServer() {
    handle = VectorsServer.start(ServerConfig.forTesting());
    client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
  }

  @AfterEach
  void stopServer() {
    if (handle != null) {
      handle.close();
    }
  }

  private URI uri(String path) {
    return URI.create("http://localhost:" + handle.port() + path);
  }

  private HttpResponse<String> postJson(String path, String body) throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(uri(path))
            .timeout(Duration.ofSeconds(10))
            .header("content-type", "application/json")
            .POST(BodyPublishers.ofString(body))
            .build();
    return client.send(req, BodyHandlers.ofString());
  }

  private HttpResponse<String> get(String path) throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(uri(path)).timeout(Duration.ofSeconds(5)).GET().build();
    return client.send(req, BodyHandlers.ofString());
  }

  private HttpResponse<String> delete(String path) throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(uri(path)).timeout(Duration.ofSeconds(5)).DELETE().build();
    return client.send(req, BodyHandlers.ofString());
  }

  @Test
  void createCollectionReturns201AndBody() throws Exception {
    String body =
        """
        {"name":"articles","dimension":8,"metric":"COSINE","indexType":"FLAT"}
        """;
    HttpResponse<String> r = postJson("/v1/collections", body);
    assertThat(r.statusCode()).isEqualTo(201);
    JsonNode n = JSON.readTree(r.body());
    assertThat(n.get("name").asText()).isEqualTo("articles");
    assertThat(n.get("dimension").asInt()).isEqualTo(8);
    assertThat(n.get("metric").asText()).isEqualTo("COSINE");
    assertThat(n.get("indexType").asText()).isEqualTo("FLAT");
    assertThat(n.get("quantizer").asText()).isEqualTo("NONE");
    assertThat(n.get("size").asInt()).isZero();
    assertThat(n.hasNonNull("createdAt")).isTrue();
  }

  @Test
  void duplicateCreateReturns409() throws Exception {
    String body =
        """
        {"name":"dup","dimension":4,"metric":"EUCLIDEAN","indexType":"FLAT"}
        """;
    assertThat(postJson("/v1/collections", body).statusCode()).isEqualTo(201);
    HttpResponse<String> r = postJson("/v1/collections", body);
    assertThat(r.statusCode()).isEqualTo(409);
    assertThat(r.headers().firstValue("content-type"))
        .hasValueSatisfying(v -> assertThat(v).startsWith("application/problem+json"));
  }

  @Test
  void invalidRequestReturns400() throws Exception {
    HttpResponse<String> r =
        postJson(
            "/v1/collections",
            "{\"name\":\"bad\",\"dimension\":-1,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}");
    assertThat(r.statusCode()).isEqualTo(400);
  }

  @Test
  void malformedJsonReturns400() throws Exception {
    HttpResponse<String> r = postJson("/v1/collections", "{not json");
    assertThat(r.statusCode()).isEqualTo(400);
  }

  @Test
  void listAndDescribeRoundTrip() throws Exception {
    postJson(
        "/v1/collections",
        "{\"name\":\"c1\",\"dimension\":4,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}");
    postJson(
        "/v1/collections",
        "{\"name\":\"c2\",\"dimension\":8,\"metric\":\"EUCLIDEAN\",\"indexType\":\"HNSW\",\"hnswM\":8}");

    HttpResponse<String> all = get("/v1/collections");
    assertThat(all.statusCode()).isEqualTo(200);
    JsonNode arr = JSON.readTree(all.body()).get("collections");
    assertThat(arr.isArray()).isTrue();
    assertThat(arr.size()).isEqualTo(2);

    HttpResponse<String> one = get("/v1/collections/c2");
    assertThat(one.statusCode()).isEqualTo(200);
    JsonNode n = JSON.readTree(one.body());
    assertThat(n.get("indexType").asText()).isEqualTo("HNSW");
    assertThat(n.get("dimension").asInt()).isEqualTo(8);
  }

  @Test
  void describeUnknownReturns404() throws Exception {
    HttpResponse<String> r = get("/v1/collections/nope");
    assertThat(r.statusCode()).isEqualTo(404);
  }

  @Test
  void dropRemovesCollection() throws Exception {
    postJson(
        "/v1/collections",
        "{\"name\":\"gone\",\"dimension\":4,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}");
    HttpResponse<String> d = delete("/v1/collections/gone");
    assertThat(d.statusCode()).isEqualTo(204);
    assertThat(get("/v1/collections/gone").statusCode()).isEqualTo(404);
    assertThat(delete("/v1/collections/gone").statusCode()).isEqualTo(404);
  }

  /**
   * Pins the contract that {@code DELETE /v1/collections/{name}} reports a 5xx problem when the
   * collection's on-disk storage cannot be fully removed — the previous behavior was a silent 204
   * with a server-side WARN log, leaving the operator unable to detect orphaned files.
   *
   * <p>The trigger is a real POSIX permission denial: a probe file is placed inside the collection
   * directory and the directory is then chmod'd to {@code 0500} (read+execute, no write), so the
   * recursive delete cannot unlink the probe child. Linux/macOS only because the trick relies on
   * POSIX file permissions.
   */
  @Test
  @EnabledOnOs({OS.LINUX, OS.MAC})
  void dropReportsErrorWhenStorageDirCannotBeFullyDeleted() throws Exception {
    // Stop the default in-memory server and start a fresh persistent one so the storage-dir branch
    // of CollectionsRoutes#drop runs.
    handle.close();
    Path dataDir = Files.createTempDirectory("vectors-drop-fail-it-");
    handle = VectorsServer.start(ServerConfig.forTesting().withDataDir(dataDir));
    try {
      String name = "stubborn";
      assertThat(
              postJson(
                      "/v1/collections",
                      "{\"name\":\""
                          + name
                          + "\",\"dimension\":4,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}")
                  .statusCode())
          .isEqualTo(201);
      Path collectionDir = dataDir.resolve(name);
      assertThat(Files.isDirectory(collectionDir)).isTrue();
      Path probe = Files.write(collectionDir.resolve("stubborn-child.tmp"), new byte[] {1, 2, 3});
      Files.setPosixFilePermissions(collectionDir, PosixFilePermissions.fromString("r-x------"));
      try {
        HttpResponse<String> r = delete("/v1/collections/" + name);
        assertThat(r.statusCode())
            .as("partial-delete must not masquerade as success")
            .isEqualTo(500);
        assertThat(r.headers().firstValue("content-type"))
            .hasValueSatisfying(v -> assertThat(v).startsWith("application/problem+json"));
        assertThat(r.body())
            .as("problem body should mention storage cleanup")
            .containsIgnoringCase("storage")
            .contains(name);
      } finally {
        // Restore perms so the temp dir can be cleaned up below.
        Files.setPosixFilePermissions(collectionDir, PosixFilePermissions.fromString("rwx------"));
        Files.deleteIfExists(probe);
      }
    } finally {
      // Best-effort cleanup of the persistent tempdir.
      if (Files.exists(dataDir)) {
        try (java.util.stream.Stream<Path> walk = Files.walk(dataDir)) {
          walk.sorted(java.util.Comparator.reverseOrder())
              .forEach(
                  p -> {
                    try {
                      Files.deleteIfExists(p);
                    } catch (Exception ignored) {
                      // best-effort
                    }
                  });
        }
      }
    }
  }
}
