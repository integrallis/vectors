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
package com.integrallis.vectors.studio.web.optimize;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.connection.EmbeddedStudioBackend;
import com.integrallis.vectors.studio.web.StudioConfig;
import com.integrallis.vectors.studio.web.StudioServer;
import com.integrallis.vectors.studio.web.StudioServerHandle;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class OptimizeApiIT {

  private static final ObjectMapper JSON = new ObjectMapper();
  private static final int DIM = 4;

  private StudioServerHandle handle;
  private HttpClient client;

  @BeforeEach
  void setUp() {
    VectorCollection c =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .build();
    for (int cluster = 0; cluster < DIM; cluster++) {
      for (int i = 0; i < 30; i++) {
        float[] v = new float[DIM];
        v[cluster] = 1.0f + (float) (i * 1e-3);
        c.add(Document.of("doc-" + cluster + "-" + i, v));
      }
    }
    c.commit();
    EmbeddedStudioBackend b = EmbeddedStudioBackend.withCollections(Map.of("docs", c));
    handle = StudioServer.start(new StudioConfig(0, new StudioSession(b)));
    client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
  }

  @AfterEach
  void tearDown() {
    if (handle != null) handle.close();
  }

  private HttpResponse<String> get(String path) throws Exception {
    return client.send(
        HttpRequest.newBuilder(URI.create("http://localhost:" + handle.port() + path))
            .timeout(Duration.ofSeconds(10))
            .GET()
            .build(),
        BodyHandlers.ofString());
  }

  private HttpResponse<String> postJson(String path, String json) throws Exception {
    return client.send(
        HttpRequest.newBuilder(URI.create("http://localhost:" + handle.port() + path))
            .timeout(Duration.ofSeconds(10))
            .header("content-type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .build(),
        BodyHandlers.ofString());
  }

  private static void waitFor(java.util.function.BooleanSupplier cond, Duration max) {
    long deadline = System.nanoTime() + max.toNanos();
    while (System.nanoTime() < deadline) {
      if (cond.getAsBoolean()) return;
      try {
        Thread.sleep(100);
      } catch (InterruptedException ie) {
        Thread.currentThread().interrupt();
        return;
      }
    }
    throw new AssertionError("timed out waiting for condition");
  }

  @Test
  void designPageRenders() throws Exception {
    HttpResponse<String> res = get("/collections/docs/optimize");
    assertThat(res.statusCode()).isEqualTo(200);
    assertThat(res.body()).contains("Optimize");
    assertThat(res.body()).contains("Sampler");
    assertThat(res.body()).contains("/static/optimize/design.js");
  }

  @Test
  void submitSseAndCancelLifecycle() throws Exception {
    String dto =
        "{\"collection\":\"docs\",\"sampler\":\"RANDOM\",\"nTrials\":3,\"kForMetrics\":5,"
            + "\"querySampleSize\":8,\"mMin\":8,\"mMax\":16,\"efMin\":50,\"efMax\":100}";
    HttpResponse<String> submit = postJson("/api/optimize/studies", dto);
    assertThat(submit.statusCode()).isEqualTo(202);
    JsonNode body = JSON.readTree(submit.body());
    String studyId = body.get("studyId").asText();

    waitFor(
        () -> {
          try {
            HttpResponse<String> r = get("/api/optimize/studies/" + studyId + "/trials");
            if (r.statusCode() != 200) return false;
            JsonNode tree = JSON.readTree(r.body());
            return "COMPLETED".equals(tree.get("state").asText()) && tree.get("trials").size() == 3;
          } catch (Exception e) {
            return false;
          }
        },
        Duration.ofSeconds(45));
  }

  @Test
  void cancelOnUnknownReturns404() throws Exception {
    HttpResponse<String> res = postJson("/api/optimize/studies/missing/cancel", "");
    assertThat(res.statusCode()).isEqualTo(404);
  }

  @Test
  void applyWithoutConfirmIsRejected() throws Exception {
    HttpResponse<String> res =
        postJson("/collections/docs/optimize/apply/study-fake/trial-fake", "");
    assertThat(res.statusCode()).isEqualTo(400);
    assertThat(res.body()).contains("confirm");
  }

  @Test
  void applyWithConfirmReturns202() throws Exception {
    HttpResponse<String> res =
        postJson("/collections/docs/optimize/apply/study-fake/trial-fake?confirm=true", "");
    assertThat(res.statusCode()).isEqualTo(202);
  }
}
