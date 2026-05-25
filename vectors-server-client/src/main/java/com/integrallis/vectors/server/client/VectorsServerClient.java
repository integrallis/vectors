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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import javax.net.ssl.SSLSession;

/**
 * HTTP client for the vectors-server REST API.
 *
 * <p>Built on {@link java.net.http.HttpClient} for zero extra dependencies. Thread-safe.
 */
public final class VectorsServerClient implements AutoCloseable {

  private static final ObjectMapper MAPPER =
      new ObjectMapper().registerModule(new JavaTimeModule());
  private static final String CONTENT_TYPE = "application/json";
  private static final Duration DEFAULT_CONNECT_TIMEOUT = Duration.ofSeconds(5);
  private static final Duration DEFAULT_REQUEST_TIMEOUT = Duration.ofSeconds(30);
  private static final long DEFAULT_MAX_RESPONSE_BYTES = 16L * 1024L * 1024L;

  private final String baseUrl;
  private final HttpClient http;
  private final Duration requestTimeout;
  private final long maxResponseBytes;

  /**
   * Creates a client connecting to the given base URL (e.g. {@code http://localhost:8287}).
   *
   * @param baseUrl the server base URL (no trailing slash)
   */
  public VectorsServerClient(String baseUrl) {
    this(baseUrl, DEFAULT_REQUEST_TIMEOUT);
  }

  /**
   * Creates a client with a per-request timeout.
   *
   * @param baseUrl the server base URL (no trailing slash)
   * @param requestTimeout timeout applied to every HTTP request
   */
  public VectorsServerClient(String baseUrl, Duration requestTimeout) {
    this(baseUrl, requestTimeout, DEFAULT_MAX_RESPONSE_BYTES);
  }

  VectorsServerClient(String baseUrl, Duration requestTimeout, long maxResponseBytes) {
    Objects.requireNonNull(baseUrl, "baseUrl");
    Objects.requireNonNull(requestTimeout, "requestTimeout");
    if (requestTimeout.isZero() || requestTimeout.isNegative()) {
      throw new IllegalArgumentException("requestTimeout must be positive");
    }
    if (maxResponseBytes < 1) {
      throw new IllegalArgumentException("maxResponseBytes must be positive");
    }
    this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
    this.requestTimeout = requestTimeout;
    this.maxResponseBytes = maxResponseBytes;
    this.http =
        HttpClient.newBuilder()
            .connectTimeout(DEFAULT_CONNECT_TIMEOUT)
            .version(HttpClient.Version.HTTP_1_1)
            .build();
  }

  // --- Collection operations ---

  /**
   * Create a new collection.
   *
   * @return collection info
   */
  public CollectionInfo createCollection(
      String name, int dimension, String metric, String indexType, Map<String, Object> params) {
    Map<String, Object> body = new HashMap<>();
    body.put("name", name);
    body.put("dimension", dimension);
    body.put("metric", metric);
    body.put("indexType", indexType);
    if (params != null) {
      body.putAll(params);
    }
    HttpResponse<String> res = post("/v1/collections", body);
    return parseJson(res.body(), CollectionInfo.class);
  }

  /** Check if a collection exists. */
  public boolean collectionExists(String name) {
    try {
      HttpResponse<String> res = get("/v1/collections/" + name);
      return res.statusCode() == 200;
    } catch (VectorsServerException e) {
      if (e.statusCode() == 404) {
        return false;
      }
      throw e;
    }
  }

  /** Delete a collection. */
  public void deleteCollection(String name) {
    delete("/v1/collections/" + name);
  }

  /** Lists all collections known to the server. */
  public List<CollectionInfo> listCollections() {
    HttpResponse<String> res = get("/v1/collections");
    try {
      JsonNode root = MAPPER.readTree(res.body());
      JsonNode arr = root.get("collections");
      if (arr == null || !arr.isArray()) return List.of();
      return MAPPER.readValue(arr.traverse(), new TypeReference<List<CollectionInfo>>() {});
    } catch (IOException e) {
      throw new RuntimeException("failed to parse listCollections response", e);
    }
  }

  /** Describes a single collection. */
  public CollectionInfo describe(String name) {
    HttpResponse<String> res = get("/v1/collections/" + name);
    return parseJson(res.body(), CollectionInfo.class);
  }

  // --- Bulk read endpoints (Studio prerequisites) ---

  /**
   * Returns a paginated preview of live documents.
   *
   * @param collection collection name
   * @param offset zero-based offset into the live document list
   * @param limit maximum number of documents to return (server caps at 10000)
   * @param includeVectors whether to include each document's vector
   */
  public DocumentPage previewDocuments(
      String collection, int offset, int limit, boolean includeVectors) {
    String path =
        "/v1/collections/"
            + collection
            + "/documents?offset="
            + offset
            + "&limit="
            + limit
            + "&includeVectors="
            + includeVectors;
    HttpResponse<String> res = get(path);
    return parseJson(res.body(), DocumentPage.class);
  }

  /**
   * Bulk fetch raw vectors by id. Missing ids are returned in {@link
   * VectorsBatchResponse#missing()}.
   */
  public VectorsBatchResponse vectorsBatch(String collection, List<String> ids) {
    HttpResponse<String> res =
        post("/v1/collections/" + collection + "/vectors-batch", Map.of("ids", ids));
    return parseJson(res.body(), VectorsBatchResponse.class);
  }

  /**
   * Returns a uniform random sample of {@code n} live documents from a collection.
   *
   * @param collection collection name
   * @param n requested sample size (server caps at 10000)
   * @param includeMetadata whether to include each document's metadata
   */
  public SampleResponse sample(String collection, int n, boolean includeMetadata) {
    HttpResponse<String> res =
        get(
            "/v1/collections/"
                + collection
                + "/sample?n="
                + n
                + "&includeMetadata="
                + includeMetadata);
    return parseJson(res.body(), SampleResponse.class);
  }

  // --- Document operations ---

  /**
   * Fetch a single live document by id.
   *
   * @return the document, or {@link Optional#empty()} if no document with that id exists
   */
  public Optional<DocumentPage.Item> getDocument(String collection, String id) {
    try {
      HttpResponse<String> res = get("/v1/collections/" + collection + "/documents/" + id);
      return Optional.of(parseJson(res.body(), DocumentPage.Item.class));
    } catch (VectorsServerException e) {
      if (e.statusCode() == 404) {
        return Optional.empty();
      }
      throw e;
    }
  }

  /**
   * Upsert documents into a collection.
   *
   * @return number of documents upserted
   */
  public int upsertDocuments(String collection, List<DocumentPayload> documents) {
    List<Map<String, Object>> docs = new ArrayList<>();
    for (DocumentPayload d : documents) {
      Map<String, Object> doc = new HashMap<>();
      doc.put("id", d.id());
      doc.put("vector", d.vector());
      if (d.text() != null) {
        doc.put("text", d.text());
      }
      if (d.metadata() != null) {
        doc.put("metadata", d.metadata());
      }
      if (d.blob() != null) {
        doc.put("blob", d.blob());
      }
      docs.add(doc);
    }
    post("/v1/collections/" + collection + "/documents", Map.of("documents", docs));
    return documents.size();
  }

  /** Delete a single document. */
  public void deleteDocument(String collection, String documentId) {
    delete("/v1/collections/" + collection + "/documents/" + documentId);
  }

  /** Explicitly commit pending changes. */
  public void commit(String collection) {
    post("/v1/collections/" + collection + "/commit", Map.of());
  }

  // --- Search operations ---

  /** Standard vector search. */
  public List<SearchHit> search(
      String collection, float[] queryVector, int k, Double minScore, Map<String, Object> filter) {
    Map<String, Object> body = new HashMap<>();
    body.put("queryVector", queryVector);
    body.put("k", k);
    body.put("includeText", true);
    body.put("includeMetadata", true);
    if (minScore != null) {
      body.put("minScore", minScore);
    }
    if (filter != null) {
      body.put("filter", filter);
    }
    HttpResponse<String> res = post("/v1/collections/" + collection + "/search", body);
    return parseSearchHits(res.body());
  }

  /** Hybrid search (vector + text, server-side RRF fusion). */
  public List<SearchHit> hybridSearch(
      String collection, float[] queryVector, String queryText, int k, String fusionMode) {
    Map<String, Object> body = new HashMap<>();
    body.put("queryVector", queryVector);
    body.put("k", k);
    body.put("includeText", true);
    body.put("includeMetadata", true);
    if (queryText != null) {
      body.put("queryText", queryText);
    }
    if (fusionMode != null) {
      body.put("hybridMode", fusionMode);
    }
    HttpResponse<String> res = post("/v1/collections/" + collection + "/search", body);
    return parseSearchHits(res.body());
  }

  /** Retrieve a blob (e.g. image bytes) by document ID. */
  public Optional<byte[]> getBlob(String collection, String documentId) {
    try {
      HttpRequest req =
          HttpRequest.newBuilder(
                  URI.create(baseUrl + "/v1/collections/" + collection + "/blobs/" + documentId))
              .timeout(requestTimeout)
              .GET()
              .build();
      HttpResponse<byte[]> res = sendBytes(req);
      if (res.statusCode() == 200) {
        return Optional.of(res.body());
      } else if (res.statusCode() == 404) {
        return Optional.empty();
      }
      throw new VectorsServerException(
          res.statusCode(), new String(res.body(), java.nio.charset.StandardCharsets.UTF_8));
    } catch (IOException | InterruptedException e) {
      if (e instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      throw new RuntimeException("failed to get blob", e);
    }
  }

  // --- Health ---

  /** Check server health. */
  public boolean isHealthy() {
    HttpResponse<String> res = get("/v1/health");
    return res.statusCode() == 200;
  }

  @Override
  public void close() {
    // HttpClient does not need explicit close in Java 21+
  }

  // --- Internal HTTP helpers ---

  private HttpResponse<String> get(String path) {
    try {
      HttpRequest req =
          HttpRequest.newBuilder(URI.create(baseUrl + path)).timeout(requestTimeout).GET().build();
      HttpResponse<String> res = sendString(req);
      if (res.statusCode() >= 400) {
        throw new VectorsServerException(res.statusCode(), res.body());
      }
      return res;
    } catch (VectorsServerException e) {
      throw e;
    } catch (IOException | InterruptedException e) {
      if (e instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      throw new RuntimeException("HTTP GET failed: " + path, e);
    }
  }

  private HttpResponse<String> post(String path, Object body) {
    try {
      HttpRequest req =
          HttpRequest.newBuilder(URI.create(baseUrl + path))
              .timeout(requestTimeout)
              .header("Content-Type", CONTENT_TYPE)
              .POST(HttpRequest.BodyPublishers.ofString(MAPPER.writeValueAsString(body)))
              .build();
      HttpResponse<String> res = sendString(req);
      if (res.statusCode() >= 400) {
        throw new VectorsServerException(res.statusCode(), res.body());
      }
      return res;
    } catch (VectorsServerException e) {
      throw e;
    } catch (IOException | InterruptedException e) {
      if (e instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      throw new RuntimeException("HTTP POST failed: " + path, e);
    }
  }

  private HttpResponse<String> delete(String path) {
    try {
      HttpRequest req =
          HttpRequest.newBuilder(URI.create(baseUrl + path))
              .timeout(requestTimeout)
              .DELETE()
              .build();
      HttpResponse<String> res = sendString(req);
      if (res.statusCode() >= 400 && res.statusCode() != 404) {
        throw new VectorsServerException(res.statusCode(), res.body());
      }
      return res;
    } catch (VectorsServerException e) {
      throw e;
    } catch (IOException | InterruptedException e) {
      if (e instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      throw new RuntimeException("HTTP DELETE failed: " + path, e);
    }
  }

  private HttpResponse<String> sendString(HttpRequest request)
      throws IOException, InterruptedException {
    HttpResponse<byte[]> response = sendBytes(request);
    return new BoundedStringResponse(response, new String(response.body(), StandardCharsets.UTF_8));
  }

  private HttpResponse<byte[]> sendBytes(HttpRequest request)
      throws IOException, InterruptedException {
    HttpResponse<InputStream> response =
        http.send(request, HttpResponse.BodyHandlers.ofInputStream());
    return new BoundedByteResponse(response, readBounded(response.body()));
  }

  private byte[] readBounded(InputStream in) throws IOException {
    try (in) {
      byte[] body = in.readNBytes(Math.toIntExact(maxResponseBytes + 1));
      if (body.length > maxResponseBytes) {
        throw new IOException("response body exceeds " + maxResponseBytes + " bytes");
      }
      return body;
    } catch (ArithmeticException e) {
      throw new IllegalStateException("maxResponseBytes is too large", e);
    }
  }

  private static <T> T parseJson(String json, Class<T> clazz) {
    try {
      return MAPPER.readValue(json, clazz);
    } catch (IOException e) {
      throw new RuntimeException("failed to parse JSON response", e);
    }
  }

  private static List<SearchHit> parseSearchHits(String json) {
    try {
      JsonNode root = MAPPER.readTree(json);
      JsonNode hits = root.get("hits");
      if (hits == null || !hits.isArray()) {
        return List.of();
      }
      return MAPPER.readValue(hits.traverse(), new TypeReference<List<SearchHit>>() {});
    } catch (IOException e) {
      throw new RuntimeException("failed to parse search response", e);
    }
  }

  private record BoundedStringResponse(HttpResponse<?> delegate, String body)
      implements HttpResponse<String> {

    @Override
    public int statusCode() {
      return delegate.statusCode();
    }

    @Override
    public HttpRequest request() {
      return delegate.request();
    }

    @Override
    public Optional<HttpResponse<String>> previousResponse() {
      return Optional.empty();
    }

    @Override
    public HttpHeaders headers() {
      return delegate.headers();
    }

    @Override
    public String body() {
      return body;
    }

    @Override
    public Optional<SSLSession> sslSession() {
      return delegate.sslSession();
    }

    @Override
    public URI uri() {
      return delegate.uri();
    }

    @Override
    public HttpClient.Version version() {
      return delegate.version();
    }
  }

  private record BoundedByteResponse(HttpResponse<?> delegate, byte[] body)
      implements HttpResponse<byte[]> {

    @Override
    public int statusCode() {
      return delegate.statusCode();
    }

    @Override
    public HttpRequest request() {
      return delegate.request();
    }

    @Override
    public Optional<HttpResponse<byte[]>> previousResponse() {
      return Optional.empty();
    }

    @Override
    public HttpHeaders headers() {
      return delegate.headers();
    }

    @Override
    public byte[] body() {
      return body;
    }

    @Override
    public Optional<SSLSession> sslSession() {
      return delegate.sslSession();
    }

    @Override
    public URI uri() {
      return delegate.uri();
    }

    @Override
    public HttpClient.Version version() {
      return delegate.version();
    }
  }
}
