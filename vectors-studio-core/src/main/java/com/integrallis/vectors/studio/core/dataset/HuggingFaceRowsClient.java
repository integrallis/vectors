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
package com.integrallis.vectors.studio.core.dataset;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * {@link DatasetLoader.RowsFetcher} backed by the HuggingFace datasets-server {@code /rows} JSON
 * API. Zero non-JDK transport dependencies: it uses {@link HttpClient} and parses the response with
 * Jackson.
 *
 * <p>The response shape is {@code
 * {"rows":[{"row_idx":N,"row":{...columns...},"truncated_cells":[]}, ...]}}; this client returns
 * the list of inner {@code "row"} maps.
 */
public final class HuggingFaceRowsClient implements DatasetLoader.RowsFetcher {

  private static final String BASE = "https://datasets-server.huggingface.co/rows";
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {};

  private final HttpClient http;
  private final String dataset;
  private final String config;
  private final String split;

  public HuggingFaceRowsClient(String dataset, String config, String split) {
    this.dataset = dataset;
    this.config = (config == null || config.isBlank()) ? "default" : config;
    this.split = (split == null || split.isBlank()) ? "train" : split;
    this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15)).build();
  }

  @Override
  @SuppressWarnings("unchecked")
  public List<Map<String, Object>> fetch(int offset, int length) throws IOException {
    String url =
        BASE
            + "?dataset="
            + enc(dataset)
            + "&config="
            + enc(config)
            + "&split="
            + enc(split)
            + "&offset="
            + offset
            + "&length="
            + length;
    HttpRequest request =
        HttpRequest.newBuilder(URI.create(url))
            .timeout(Duration.ofSeconds(60))
            .header("Accept", "application/json")
            .GET()
            .build();
    HttpResponse<String> response;
    try {
      response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("interrupted while fetching dataset rows", e);
    }
    if (response.statusCode() / 100 != 2) {
      throw new IOException(
          "datasets-server returned HTTP " + response.statusCode() + ": " + response.body());
    }
    Map<String, Object> parsed = MAPPER.readValue(response.body(), MAP_TYPE);
    Object rowsObj = parsed.get("rows");
    if (!(rowsObj instanceof List<?> rows)) {
      return List.of();
    }
    List<Map<String, Object>> out = new ArrayList<>(rows.size());
    for (Object entry : rows) {
      if (entry instanceof Map<?, ?> wrapper) {
        Object inner = wrapper.get("row");
        if (inner instanceof Map<?, ?> row) {
          out.add((Map<String, Object>) row);
        }
      }
    }
    return out;
  }

  private static String enc(String s) {
    return URLEncoder.encode(s, StandardCharsets.UTF_8);
  }
}
