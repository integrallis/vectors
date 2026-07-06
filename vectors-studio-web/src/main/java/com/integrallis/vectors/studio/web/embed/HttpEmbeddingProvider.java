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
package com.integrallis.vectors.studio.web.embed;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.Objects;

/**
 * {@link EmbeddingProvider} that calls an OpenAI-compatible {@code POST {baseUrl}/embeddings}
 * endpoint over the JDK {@link HttpClient}. Sends {@code {"input", "model"[, "dimensions"]}} and
 * parses {@code data[0].embedding} into a {@code float[]}. An {@code Authorization: Bearer} header
 * is added only when a non-blank API key is supplied.
 */
public final class HttpEmbeddingProvider implements EmbeddingProvider {

  private static final int SNIPPET_LIMIT = 500;

  private final String embeddingsUrl;
  private final String apiKey;
  private final String model;
  private final Integer dimensions;
  private final ObjectMapper mapper;
  private final HttpClient http;

  /**
   * @param baseUrl API base URL, e.g. {@code https://api.openai.com/v1}
   * @param apiKey bearer token, or {@code null}/blank for no authentication
   * @param model embedding model id
   * @param dimensions requested output dimensionality, or {@code null} for the model default
   * @param mapper shared JSON mapper
   */
  public HttpEmbeddingProvider(
      String baseUrl, String apiKey, String model, Integer dimensions, ObjectMapper mapper) {
    Objects.requireNonNull(baseUrl, "baseUrl");
    String trimmed = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
    this.embeddingsUrl = trimmed + "/embeddings";
    this.apiKey = apiKey;
    this.model = Objects.requireNonNull(model, "model");
    this.dimensions = dimensions;
    this.mapper = Objects.requireNonNull(mapper, "mapper");
    this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
  }

  @Override
  public float[] embed(String text) {
    try {
      ObjectNode body = mapper.createObjectNode();
      body.put("input", text == null ? "" : text);
      body.put("model", model);
      if (dimensions != null) {
        body.put("dimensions", dimensions.intValue());
      }
      HttpRequest.Builder rb =
          HttpRequest.newBuilder(URI.create(embeddingsUrl))
              .timeout(Duration.ofSeconds(30))
              .header("Content-Type", "application/json")
              .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)));
      if (apiKey != null && !apiKey.isBlank()) {
        rb.header("Authorization", "Bearer " + apiKey);
      }
      HttpResponse<String> res = http.send(rb.build(), BodyHandlers.ofString());
      if (res.statusCode() / 100 != 2) {
        throw new IOException(
            "embedding request to "
                + embeddingsUrl
                + " for model "
                + model
                + " failed with HTTP "
                + res.statusCode()
                + ": "
                + snippet(res.body()));
      }
      JsonNode embedding = mapper.readTree(res.body()).path("data").path(0).path("embedding");
      if (!embedding.isArray() || embedding.isEmpty()) {
        throw new IOException(
            "embedding response from "
                + embeddingsUrl
                + " had no data[0].embedding array: "
                + snippet(res.body()));
      }
      float[] out = new float[embedding.size()];
      for (int i = 0; i < out.length; i++) {
        out[i] = (float) embedding.get(i).asDouble();
      }
      return out;
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("embedding request interrupted", e);
    }
  }

  private static String snippet(String body) {
    if (body == null) {
      return "<empty>";
    }
    return body.length() <= SNIPPET_LIMIT ? body : body.substring(0, SNIPPET_LIMIT) + "…";
  }
}
