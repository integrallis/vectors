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
package com.integrallis.vectors.studio.sidecart.sources;

import com.integrallis.vectors.studio.sidecart.SidecartSourceException;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Thin HTTP shim over Cloudflare's D1 {@code /query} endpoint. Encapsulates URL construction,
 * Bearer-token auth, JSON request shape, and response unwrapping so {@link D1SidecartSource} and
 * {@link D1SidecartWriter} can stay focused on SQL.
 */
final class D1HttpClient implements AutoCloseable {

  private static final String BASE_URL = "https://api.cloudflare.com/client/v4";
  private static final Duration TIMEOUT = Duration.ofSeconds(30);
  private static final int DEFAULT_MAX_ATTEMPTS = 3;
  private static final Duration DEFAULT_BACKOFF = Duration.ofMillis(100);

  private final HttpClient http;
  private final URI queryUri;
  private final String authHeader;
  private final int maxAttempts;
  private final Duration baseBackoff;
  private final Sleeper sleeper;

  D1HttpClient(HttpClient http, String accountId, String databaseId, String apiToken) {
    this(
        http,
        accountId,
        databaseId,
        apiToken,
        DEFAULT_MAX_ATTEMPTS,
        DEFAULT_BACKOFF,
        Thread::sleep);
  }

  D1HttpClient(
      HttpClient http,
      String accountId,
      String databaseId,
      String apiToken,
      int maxAttempts,
      Duration baseBackoff,
      Sleeper sleeper) {
    if (maxAttempts <= 0) throw new IllegalArgumentException("maxAttempts must be positive");
    this.baseBackoff = Objects.requireNonNull(baseBackoff, "baseBackoff");
    if (baseBackoff.isNegative()) throw new IllegalArgumentException("baseBackoff must be >= 0");
    this.http = Objects.requireNonNull(http, "http");
    this.queryUri =
        URI.create(BASE_URL + "/accounts/" + accountId + "/d1/database/" + databaseId + "/query");
    this.authHeader = "Bearer " + apiToken;
    this.maxAttempts = maxAttempts;
    this.sleeper = Objects.requireNonNull(sleeper, "sleeper");
  }

  /**
   * Executes a single parameterised SQL statement and returns its result rows. Each row is a map
   * keyed by the column name as it appears in the response (lower-cased by D1 unless the SQL
   * specified an alias).
   */
  List<Map<String, Object>> queryRows(String sql, List<?> params) {
    Map<String, Object> envelope = postQuery(sql, params);
    List<Map<String, Object>> result = unwrapResults(envelope);
    if (result.isEmpty()) return List.of();
    Object rows = result.get(0).get("results");
    if (!(rows instanceof List<?> rl)) return List.of();
    List<Map<String, Object>> out = new java.util.ArrayList<>(rl.size());
    for (Object o : rl) {
      if (o instanceof Map<?, ?> m) {
        Map<String, Object> typed = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : m.entrySet()) typed.put(String.valueOf(e.getKey()), e.getValue());
        out.add(typed);
      }
    }
    return out;
  }

  /** Executes one or more SQL statements, ignoring the row payload. Used for DDL / writes. */
  void execute(String sql, List<?> params) {
    postQuery(sql, params);
  }

  /**
   * Executes a list of parameterised statements in one HTTP round-trip. The Cloudflare D1 {@code
   * /query} endpoint accepts either a single {@code {sql, params}} object or a {@code {batch:
   * [{sql, params}, …]}} envelope; this method uses the latter shape.
   */
  void executeBatch(List<Map<String, Object>> statements) {
    Map<String, Object> envelope = new LinkedHashMap<>();
    envelope.put("batch", statements);
    String body = D1Json.encode(envelope);
    HttpRequest request =
        HttpRequest.newBuilder(queryUri)
            .timeout(TIMEOUT)
            .header("Authorization", authHeader)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();
    sendAndCheck(request);
  }

  private Map<String, Object> postQuery(String sql, List<?> params) {
    String body = D1SidecartSource.buildRequestBody(sql, params);
    HttpRequest request =
        HttpRequest.newBuilder(queryUri)
            .timeout(TIMEOUT)
            .header("Authorization", authHeader)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();
    return sendAndCheck(request);
  }

  @SuppressWarnings("unchecked")
  private Map<String, Object> sendAndCheck(HttpRequest request) {
    HttpResponse<String> response = null;
    int attempts = 0;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      attempts = attempt;
      try {
        response = http.send(request, HttpResponse.BodyHandlers.ofString());
      } catch (IOException e) {
        throw new SidecartSourceException("D1 request failed: " + e.getMessage(), e);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new SidecartSourceException("D1 request interrupted", e);
      }
      if (!shouldRetry(response.statusCode()) || attempt == maxAttempts) break;
      sleepBeforeRetry(attempt, response.statusCode());
    }
    if (response == null) throw new SidecartSourceException("D1 request was not attempted");
    if (response.statusCode() / 100 != 2) {
      throw new SidecartSourceException(
          "D1 returned HTTP "
              + response.statusCode()
              + " after "
              + attempts
              + " attempt(s): "
              + response.body(),
          null);
    }
    Map<String, Object> envelope = D1Json.decodeObject(response.body());
    Object success = envelope.get("success");
    if (!Boolean.TRUE.equals(success)) {
      throw new SidecartSourceException("D1 reported failure: " + response.body(), null);
    }
    return envelope;
  }

  private void sleepBeforeRetry(int failedAttempt, int statusCode) {
    long delayMillis = retryDelayMillis(failedAttempt);
    try {
      sleeper.sleep(delayMillis);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new SidecartSourceException("D1 retry interrupted after HTTP " + statusCode, e);
    }
  }

  private long retryDelayMillis(int failedAttempt) {
    long baseMillis = baseBackoff.toMillis();
    if (baseMillis == 0) return 0;
    int exponent = Math.min(failedAttempt - 1, 30);
    long exponential = baseMillis * (1L << exponent);
    long jitter = ThreadLocalRandom.current().nextLong(baseMillis + 1);
    return exponential + jitter;
  }

  private static boolean shouldRetry(int statusCode) {
    return statusCode == 429 || statusCode / 100 == 5;
  }

  @SuppressWarnings("unchecked")
  private static List<Map<String, Object>> unwrapResults(Map<String, Object> envelope) {
    Object result = envelope.get("result");
    if (result instanceof List<?> list) {
      List<Map<String, Object>> out = new java.util.ArrayList<>(list.size());
      for (Object o : list) {
        if (o instanceof Map<?, ?> m) out.add((Map<String, Object>) m);
      }
      return out;
    }
    return Collections.emptyList();
  }

  @Override
  public void close() {
    // The shared HttpClient is owned by whoever passed it in.
  }

  @FunctionalInterface
  interface Sleeper {
    void sleep(long millis) throws InterruptedException;
  }
}
