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
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.Random;
import org.junit.jupiter.api.Test;

class HnswEndToEndTest {

  private static final ObjectMapper JSON = ObjectMapperHolder.shared();
  private static final int DIM = 32;
  private static final int N = 1000;

  @Test
  void ingestAndSearch1000DocsHnsw() throws Exception {
    try (VectorsServer.ServerHandle handle = VectorsServer.start(ServerConfig.forTesting())) {
      HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
      int port = handle.port();

      // Create a 32-D HNSW COSINE collection.
      String create =
          "{\"name\":\"big\",\"dimension\":"
              + DIM
              + ",\"metric\":\"COSINE\",\"indexType\":\"HNSW\",\"hnswM\":8,\"hnswEfConstruction\":64}";
      HttpResponse<String> r = postJson(client, port, "/v1/collections", create);
      assertThat(r.statusCode()).isEqualTo(201);

      // Generate N=1000 deterministic random unit vectors in DIM=32 dims and bulk-upsert.
      Random rng = new Random(42);
      ObjectNode body = JsonNodeFactory.instance.objectNode();
      ArrayNode docs = body.putArray("documents");
      float[][] all = new float[N][DIM];
      for (int i = 0; i < N; i++) {
        float[] v = randomUnit(rng);
        all[i] = v;
        ObjectNode d = docs.addObject();
        d.put("id", "doc-" + i);
        ArrayNode arr = d.putArray("vector");
        for (float x : v) {
          arr.add(x);
        }
      }
      HttpResponse<String> up =
          postJson(client, port, "/v1/collections/big/documents", JSON.writeValueAsString(body));
      assertThat(up.statusCode()).isEqualTo(200);
      JsonNode upBody = JSON.readTree(up.body());
      assertThat(upBody.get("upserted").asInt()).isEqualTo(N);
      assertThat(upBody.get("size").asInt()).isEqualTo(N);

      // Query with one of the stored vectors — its own id must be top-1 with score ~1.0.
      int queryIdx = 137;
      ObjectNode q = JsonNodeFactory.instance.objectNode();
      ArrayNode qv = q.putArray("queryVector");
      for (float x : all[queryIdx]) {
        qv.add(x);
      }
      q.put("k", 5);
      q.put("includeVector", false);

      HttpResponse<String> sr =
          postJson(client, port, "/v1/collections/big/search", JSON.writeValueAsString(q));
      assertThat(sr.statusCode()).isEqualTo(200);
      JsonNode hits = JSON.readTree(sr.body()).get("hits");
      assertThat(hits.size()).isEqualTo(5);
      assertThat(hits.get(0).get("id").asText()).isEqualTo("doc-" + queryIdx);
      assertThat(hits.get(0).get("score").asDouble()).isGreaterThan(0.999);
    }
  }

  private static float[] randomUnit(Random rng) {
    float[] v = new float[DIM];
    double sum = 0;
    for (int i = 0; i < DIM; i++) {
      v[i] = (float) rng.nextGaussian();
      sum += v[i] * v[i];
    }
    float norm = (float) Math.sqrt(sum);
    if (norm > 0) {
      for (int i = 0; i < DIM; i++) {
        v[i] /= norm;
      }
    }
    return v;
  }

  private static HttpResponse<String> postJson(
      HttpClient client, int port, String path, String body) throws Exception {
    HttpRequest req =
        HttpRequest.newBuilder(URI.create("http://localhost:" + port + path))
            .timeout(Duration.ofSeconds(30))
            .header("content-type", "application/json")
            .POST(BodyPublishers.ofString(body))
            .build();
    return client.send(req, BodyHandlers.ofString());
  }
}
