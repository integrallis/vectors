package com.integrallis.vectors.demo.serverclient;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.server.ServerConfig;
import com.integrallis.vectors.server.VectorsServer;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * End-to-end HTTP round-trip demo.
 *
 * <p>Starts a {@link VectorsServer} on an OS-assigned ephemeral port, creates a collection, upserts
 * a handful of documents, runs a k-NN search, and shuts the server down cleanly. No external tools
 * required — the demo is self-contained and exercises the same wire protocol a production client
 * would use.
 *
 * <p>Run:
 *
 * <pre>
 *   ./gradlew :demos:server-client:run
 * </pre>
 */
public final class ServerClientApp {

  private static final int DIMENSION = 16;
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private ServerClientApp() {}

  public static void main(String[] args) throws Exception {
    Path dataDir = Files.createTempDirectory("java-vectors-demo-server-client-");
    ServerConfig config = ServerConfig.forTesting().withDataDir(dataDir);

    try (VectorsServer.ServerHandle handle = VectorsServer.start(config)) {
      int port = handle.port();
      String baseUrl = "http://127.0.0.1:" + port;
      System.out.printf("server: %s  (data dir: %s)%n", baseUrl, dataDir);

      HttpClient http =
          HttpClient.newBuilder()
              .connectTimeout(Duration.ofSeconds(2))
              .version(HttpClient.Version.HTTP_1_1)
              .build();

      createCollection(http, baseUrl);
      upsertDocuments(http, baseUrl);
      searchAndPrint(http, baseUrl);
      describeCollection(http, baseUrl);
      dropCollection(http, baseUrl);
    } finally {
      deleteQuietly(dataDir);
    }
  }

  private static void createCollection(HttpClient http, String baseUrl) throws Exception {
    Map<String, Object> body =
        Map.of(
            "name", "demo",
            "dimension", DIMENSION,
            "metric", "COSINE",
            "indexType", "FLAT",
            "autoCommitThreshold", 1);
    HttpResponse<String> res = post(http, baseUrl + "/v1/collections", body);
    System.out.printf("POST /v1/collections -> %d%n", res.statusCode());
    expect(res, 201, 200);
  }

  private static void upsertDocuments(HttpClient http, String baseUrl) throws Exception {
    Random rnd = new Random(42L);
    List<Map<String, Object>> docs =
        List.of(
            doc("doc-1", unit(DIMENSION, rnd), "What is HNSW?"),
            doc("doc-2", unit(DIMENSION, rnd), "Explain product quantization."),
            doc("doc-3", unit(DIMENSION, rnd), "How does mmap persistence work?"),
            doc("doc-4", unit(DIMENSION, rnd), "Describe scalar quantization."),
            doc("doc-5", unit(DIMENSION, rnd), "Why is SIMD fast on the JVM?"));
    HttpResponse<String> res =
        post(http, baseUrl + "/v1/collections/demo/documents", Map.of("documents", docs));
    System.out.printf(
        "POST /v1/collections/demo/documents -> %d  %s%n", res.statusCode(), res.body());
    expect(res, 200);
  }

  private static void searchAndPrint(HttpClient http, String baseUrl) throws Exception {
    Random rnd = new Random(7L);
    Map<String, Object> body =
        Map.of("queryVector", unit(DIMENSION, rnd), "k", 3, "includeText", true);
    HttpResponse<String> res = post(http, baseUrl + "/v1/collections/demo/search", body);
    System.out.printf("POST /v1/collections/demo/search -> %d%n", res.statusCode());
    expect(res, 200);
    JsonNode response = MAPPER.readTree(res.body());
    System.out.println("top-3 hits:");
    for (JsonNode hit : response.get("hits")) {
      System.out.printf(
          "  id=%-6s  score=%.4f  text=%s%n",
          hit.get("id").asText(), hit.get("score").asDouble(), hit.get("text").asText());
    }
  }

  private static void describeCollection(HttpClient http, String baseUrl) throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(URI.create(baseUrl + "/v1/collections/demo")).GET().build();
    HttpResponse<String> res = http.send(req, BodyHandlers.ofString());
    System.out.printf("GET  /v1/collections/demo -> %d  %s%n", res.statusCode(), res.body());
    expect(res, 200);
  }

  private static void dropCollection(HttpClient http, String baseUrl) throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(URI.create(baseUrl + "/v1/collections/demo")).DELETE().build();
    HttpResponse<String> res = http.send(req, BodyHandlers.ofString());
    System.out.printf("DELETE /v1/collections/demo -> %d%n", res.statusCode());
    expect(res, 204, 200);
  }

  private static HttpResponse<String> post(HttpClient http, String url, Object body)
      throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(URI.create(url))
            .header("Content-Type", "application/json")
            .POST(BodyPublishers.ofString(MAPPER.writeValueAsString(body)))
            .build();
    return http.send(req, BodyHandlers.ofString());
  }

  private static Map<String, Object> doc(String id, float[] v, String text) {
    return Map.of("id", id, "vector", v, "text", text);
  }

  private static float[] unit(int dim, Random rnd) {
    float[] v = new float[dim];
    float norm = 0f;
    for (int i = 0; i < dim; i++) {
      v[i] = (float) rnd.nextGaussian();
      norm += v[i] * v[i];
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0f) {
      for (int i = 0; i < dim; i++) v[i] /= norm;
    }
    return v;
  }

  private static void expect(HttpResponse<String> res, int... expected) {
    for (int e : expected) if (res.statusCode() == e) return;
    throw new IllegalStateException("unexpected status " + res.statusCode() + ": " + res.body());
  }

  private static void deleteQuietly(Path dir) {
    try {
      if (!Files.exists(dir)) return;
      try (var stream = Files.walk(dir)) {
        stream
            .sorted((a, b) -> b.getNameCount() - a.getNameCount())
            .forEach(p -> p.toFile().delete());
      }
    } catch (Exception ignored) {
    }
  }
}
