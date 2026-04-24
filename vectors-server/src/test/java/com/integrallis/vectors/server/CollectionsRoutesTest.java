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
}
