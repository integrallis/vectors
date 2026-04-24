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

class DocumentsAndSearchTest {

  private static final ObjectMapper JSON = ObjectMapperHolder.shared();

  private VectorsServer.ServerHandle handle;
  private HttpClient client;

  @BeforeEach
  void setUp() throws Exception {
    handle = VectorsServer.start(ServerConfig.forTesting());
    client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    // Create a small 4-D COSINE FLAT collection up front.
    HttpResponse<String> r =
        postJson(
            "/v1/collections",
            "{\"name\":\"docs\",\"dimension\":4,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}");
    assertThat(r.statusCode()).isEqualTo(201);
  }

  @AfterEach
  void tearDown() {
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

  private HttpResponse<String> delete(String path) throws Exception {
    return client.send(
        HttpRequest.newBuilder(uri(path)).timeout(Duration.ofSeconds(5)).DELETE().build(),
        BodyHandlers.ofString());
  }

  @Test
  void upsertBatchAndSearchRoundTrip() throws Exception {
    String batch =
        """
        {"documents":[
          {"id":"a","vector":[1.0,0.0,0.0,0.0],"text":"aye","metadata":{"cat":"x","rank":1}},
          {"id":"b","vector":[0.0,1.0,0.0,0.0],"text":"bee","metadata":{"cat":"y","rank":2}},
          {"id":"c","vector":[0.0,0.0,1.0,0.0],"text":"cee","metadata":{"cat":"x","rank":3}}
        ]}
        """;
    HttpResponse<String> up = postJson("/v1/collections/docs/documents", batch);
    assertThat(up.statusCode()).isEqualTo(200);
    JsonNode upBody = JSON.readTree(up.body());
    assertThat(upBody.get("upserted").asInt()).isEqualTo(3);
    assertThat(upBody.get("size").asInt()).isEqualTo(3);

    // Nearest neighbour of [1,0,0,0] is 'a' with cosine score 1.0.
    HttpResponse<String> sr =
        postJson(
            "/v1/collections/docs/search",
            "{\"queryVector\":[1.0,0.0,0.0,0.0],\"k\":3,\"includeVector\":false}");
    assertThat(sr.statusCode()).isEqualTo(200);
    JsonNode hits = JSON.readTree(sr.body()).get("hits");
    assertThat(hits.size()).isEqualTo(3);
    assertThat(hits.get(0).get("id").asText()).isEqualTo("a");
    assertThat(hits.get(0).get("score").asDouble()).isGreaterThan(0.99);
    assertThat(hits.get(0).get("text").asText()).isEqualTo("aye");
    assertThat(hits.get(0).get("metadata").get("cat").asText()).isEqualTo("x");
    assertThat(hits.get(0).hasNonNull("vector")).isFalse();
  }

  @Test
  void badDimensionReturns400() throws Exception {
    HttpResponse<String> r =
        postJson(
            "/v1/collections/docs/documents",
            "{\"documents\":[{\"id\":\"bad\",\"vector\":[1.0,0.0,0.0]}]}");
    assertThat(r.statusCode()).isEqualTo(400);
  }

  @Test
  void missingVectorReturns400() throws Exception {
    HttpResponse<String> r =
        postJson("/v1/collections/docs/documents", "{\"documents\":[{\"id\":\"no-vec\"}]}");
    assertThat(r.statusCode()).isEqualTo(400);
  }

  @Test
  void documentsForUnknownCollectionReturns404() throws Exception {
    HttpResponse<String> r = postJson("/v1/collections/missing/documents", "{\"documents\":[]}");
    assertThat(r.statusCode()).isEqualTo(404);
  }

  @Test
  void deleteDocumentRoundTrip() throws Exception {
    postJson(
        "/v1/collections/docs/documents",
        "{\"documents\":[{\"id\":\"gone\",\"vector\":[1.0,0.0,0.0,0.0]}]}");
    assertThat(delete("/v1/collections/docs/documents/gone").statusCode()).isEqualTo(204);
    assertThat(delete("/v1/collections/docs/documents/gone").statusCode()).isEqualTo(404);
  }

  @Test
  void searchUnknownCollectionReturns404() throws Exception {
    HttpResponse<String> r =
        postJson("/v1/collections/missing/search", "{\"queryVector\":[1.0,0.0,0.0,0.0],\"k\":3}");
    assertThat(r.statusCode()).isEqualTo(404);
  }

  @Test
  void malformedFilterReturns400() throws Exception {
    HttpResponse<String> r =
        postJson(
            "/v1/collections/docs/search",
            "{\"queryVector\":[1,0,0,0],\"k\":3,\"filter\":{\"field\":\"cat\",\"foo\":\"x\"}}");
    assertThat(r.statusCode()).isEqualTo(400);
    assertThat(r.body()).contains("invalid filter");
  }

  @Test
  void commitEndpointIsIdempotent() throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(uri("/v1/collections/docs/commit"))
            .timeout(Duration.ofSeconds(5))
            .POST(BodyPublishers.noBody())
            .build();
    assertThat(client.send(req, BodyHandlers.ofString()).statusCode()).isEqualTo(204);
    assertThat(client.send(req, BodyHandlers.ofString()).statusCode()).isEqualTo(204);
  }

  @Test
  void tagsMetadataRoundTrip() throws Exception {
    postJson(
        "/v1/collections/docs/documents",
        "{\"documents\":[{\"id\":\"tagged\",\"vector\":[0.0,0.0,0.0,1.0],\"metadata\":{\"tags\":[\"ai\",\"ml\"]}}]}");
    HttpResponse<String> sr =
        postJson("/v1/collections/docs/search", "{\"queryVector\":[0.0,0.0,0.0,1.0],\"k\":1}");
    JsonNode tags = JSON.readTree(sr.body()).get("hits").get(0).get("metadata").get("tags");
    assertThat(tags.isArray()).isTrue();
    assertThat(tags.size()).isEqualTo(2);
    assertThat(tags.get(0).asText()).isEqualTo("ai");
  }

  @Test
  void filteredSearchOnlyReturnsMatchingDocs() throws Exception {
    postJson(
        "/v1/collections/docs/documents",
        "{\"documents\":[{\"id\":\"x1\",\"vector\":[1.0,0.0,0.0,0.0],\"metadata\":{\"cat\":\"x\",\"rank\":1}},"
            + "{\"id\":\"y1\",\"vector\":[1.0,0.0,0.0,0.0],\"metadata\":{\"cat\":\"y\",\"rank\":2}},"
            + "{\"id\":\"x2\",\"vector\":[0.9,0.1,0.0,0.0],\"metadata\":{\"cat\":\"x\",\"rank\":5}}]}");

    HttpResponse<String> sr =
        postJson(
            "/v1/collections/docs/search",
            "{\"queryVector\":[1.0,0.0,0.0,0.0],\"k\":10,"
                + "\"filter\":{\"field\":\"cat\",\"eq\":\"x\"}}");
    assertThat(sr.statusCode()).isEqualTo(200);
    JsonNode hits = JSON.readTree(sr.body()).get("hits");
    assertThat(hits.size()).isEqualTo(2);
    for (JsonNode h : hits) {
      assertThat(h.get("id").asText()).startsWith("x");
    }

    HttpResponse<String> sr2 =
        postJson(
            "/v1/collections/docs/search",
            "{\"queryVector\":[1.0,0.0,0.0,0.0],\"k\":10,"
                + "\"filter\":{\"and\":[{\"field\":\"cat\",\"eq\":\"x\"},{\"field\":\"rank\",\"gte\":3}]}}");
    JsonNode hits2 = JSON.readTree(sr2.body()).get("hits");
    assertThat(hits2.size()).isEqualTo(1);
    assertThat(hits2.get(0).get("id").asText()).isEqualTo("x2");
  }
}
