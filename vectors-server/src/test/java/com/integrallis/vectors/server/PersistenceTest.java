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
import java.time.Duration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PersistenceTest {

  private static final ObjectMapper JSON = ObjectMapperHolder.shared();

  private static ServerConfig persistentConfig(Path dataDir) {
    return new ServerConfig(0, dataDir.toAbsolutePath(), 64, 5);
  }

  private static HttpResponse<String> postJson(
      HttpClient client, int port, String path, String body) throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(URI.create("http://localhost:" + port + path))
            .timeout(Duration.ofSeconds(10))
            .header("content-type", "application/json")
            .POST(BodyPublishers.ofString(body))
            .build();
    return client.send(req, BodyHandlers.ofString());
  }

  private static HttpResponse<String> get(HttpClient client, int port, String path)
      throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(URI.create("http://localhost:" + port + path))
            .timeout(Duration.ofSeconds(5))
            .GET()
            .build();
    return client.send(req, BodyHandlers.ofString());
  }

  private static HttpResponse<String> delete(HttpClient client, int port, String path)
      throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(URI.create("http://localhost:" + port + path))
            .timeout(Duration.ofSeconds(5))
            .DELETE()
            .build();
    return client.send(req, BodyHandlers.ofString());
  }

  @Test
  void createSurvivesRestart(@TempDir Path dataDir) throws Exception {
    HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    String body =
        """
        {"name":"persisted","dimension":4,"metric":"COSINE","indexType":"FLAT"}
        """;

    // Phase 1: create, then stop.
    try (VectorsServer.ServerHandle h1 = VectorsServer.start(persistentConfig(dataDir))) {
      assertThat(postJson(client, h1.port(), "/v1/collections", body).statusCode()).isEqualTo(201);
      assertThat(Files.isDirectory(dataDir.resolve("persisted"))).isTrue();
    }

    // Phase 2: restart with the same dataDir — the collection should be rediscovered.
    try (VectorsServer.ServerHandle h2 = VectorsServer.start(persistentConfig(dataDir))) {
      HttpResponse<String> list = get(client, h2.port(), "/v1/collections");
      assertThat(list.statusCode()).isEqualTo(200);
      JsonNode arr = JSON.readTree(list.body()).get("collections");
      assertThat(arr.size()).isEqualTo(1);
      assertThat(arr.get(0).get("name").asText()).isEqualTo("persisted");
      assertThat(arr.get(0).get("dimension").asInt()).isEqualTo(4);
      assertThat(arr.get(0).get("metric").asText()).isEqualTo("COSINE");
      assertThat(arr.get(0).get("indexType").asText()).isEqualTo("FLAT");

      // Describe also works.
      HttpResponse<String> desc = get(client, h2.port(), "/v1/collections/persisted");
      assertThat(desc.statusCode()).isEqualTo(200);
    }
  }

  @Test
  void dropRemovesStorageDirectory(@TempDir Path dataDir) throws Exception {
    HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    try (VectorsServer.ServerHandle h = VectorsServer.start(persistentConfig(dataDir))) {
      assertThat(
              postJson(
                      client,
                      h.port(),
                      "/v1/collections",
                      "{\"name\":\"doomed\",\"dimension\":4,\"metric\":\"EUCLIDEAN\",\"indexType\":\"FLAT\"}")
                  .statusCode())
          .isEqualTo(201);
      assertThat(Files.isDirectory(dataDir.resolve("doomed"))).isTrue();

      assertThat(delete(client, h.port(), "/v1/collections/doomed").statusCode()).isEqualTo(204);
      assertThat(Files.exists(dataDir.resolve("doomed"))).isFalse();
    }
  }

  @Test
  void strayFileInDataDirIsIgnored(@TempDir Path dataDir) throws Exception {
    Files.writeString(dataDir.resolve("stray.txt"), "not a collection");
    Files.createDirectory(dataDir.resolve("empty-dir"));
    Files.createDirectory(dataDir.resolve("bad name with spaces"));

    HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    try (VectorsServer.ServerHandle h = VectorsServer.start(persistentConfig(dataDir))) {
      HttpResponse<String> list = get(client, h.port(), "/v1/collections");
      assertThat(list.statusCode()).isEqualTo(200);
      assertThat(JSON.readTree(list.body()).get("collections").size()).isZero();
    }
  }
}
