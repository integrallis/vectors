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
  private EmbeddedStudioBackend backend;

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
    backend = EmbeddedStudioBackend.withCollections(Map.of("docs", c));
    handle = StudioServer.start(new StudioConfig(0, new StudioSession(backend)));
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
  void submitCompletesAndListsTrials() throws Exception {
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

  // ---- apply tests ----
  //
  // The old applyWithConfirmReturns202 IT (audit §HIGH#5) asserted 202 against a fake study/trial.
  // It was a passing test that disguised an unimplemented endpoint. The new tests below drive a
  // real study to completion, pick a real trial, hit apply against the real collection, and check
  // that the live collection's index was actually rebuilt with the trial's parameters.

  @Test
  void applyWithoutConfirmIsRejected() throws Exception {
    HttpResponse<String> res =
        postJson("/collections/docs/optimize/apply/study-fake/trial-fake", "");
    assertThat(res.statusCode()).isEqualTo(400);
    assertThat(res.body()).containsIgnoringCase("confirm=true");
  }

  @Test
  void applyOnUnknownStudyReturns404() throws Exception {
    HttpResponse<String> res =
        postJson("/collections/docs/optimize/apply/study-fake/trial-fake?confirm=true", "");
    assertThat(res.statusCode()).isEqualTo(404);
    assertThat(res.body()).contains("study-fake");
  }

  @Test
  void applyRebuildsLiveCollectionWithTrialParameters() throws Exception {
    String dto =
        "{\"collection\":\"docs\",\"sampler\":\"RANDOM\",\"nTrials\":3,\"kForMetrics\":5,"
            + "\"querySampleSize\":8,\"mMin\":8,\"mMax\":16,\"efMin\":50,\"efMax\":100}";
    HttpResponse<String> submit = postJson("/api/optimize/studies", dto);
    assertThat(submit.statusCode()).isEqualTo(202);
    String studyId = JSON.readTree(submit.body()).get("studyId").asText();

    waitFor(
        () -> {
          try {
            HttpResponse<String> r = get("/api/optimize/studies/" + studyId + "/trials");
            if (r.statusCode() != 200) return false;
            JsonNode tree = JSON.readTree(r.body());
            return "COMPLETED".equals(tree.get("state").asText()) && tree.get("trials").size() >= 1;
          } catch (Exception e) {
            return false;
          }
        },
        Duration.ofSeconds(45));

    HttpResponse<String> trialsRes = get("/api/optimize/studies/" + studyId + "/trials");
    JsonNode trials = JSON.readTree(trialsRes.body()).get("trials");
    assertThat(trials.size()).isPositive();
    String firstTrialId = trials.get(0).get("trial").get("trialId").asText();

    HttpResponse<String> applyRes =
        postJson(
            "/collections/docs/optimize/apply/" + studyId + "/" + firstTrialId + "?confirm=true",
            "");
    assertThat(applyRes.statusCode())
        .as("apply against a real study/trial must succeed")
        .isEqualTo(200);
    JsonNode body = JSON.readTree(applyRes.body());
    assertThat(body.get("collection").asText()).isEqualTo("docs");
    assertThat(body.get("studyId").asText()).isEqualTo(studyId);
    assertThat(body.get("trialId").asText()).isEqualTo(firstTrialId);
    assertThat(body.get("documentsCopied").asInt())
        .as("the rebuild must carry every live document into the new collection (we seeded 120)")
        .isEqualTo(120);
    JsonNode applied = body.get("appliedParams");
    assertThat(applied).isNotNull();
    assertThat(applied.get("indexType").asText())
        .as("the random sampler's space is HNSW with m/efConstruction axes")
        .isEqualTo("HNSW");
    assertThat(applied.has("m")).isTrue();
    assertThat(applied.has("efConstruction")).isTrue();

    // And the live collection's *current* config must reflect the rebuild: a direct backend
    // describe should now report HNSW (we seeded the collection as FLAT in @BeforeEach).
    assertThat(backend.describe("docs").indexType())
        .as("live collection must now be HNSW (was FLAT before apply)")
        .isEqualTo("HNSW");
    assertThat(backend.describe("docs").size())
        .as("rebuild preserves every live document")
        .isEqualTo(120);
  }

  @Test
  void applyWithUnknownTrialReturns404() throws Exception {
    String dto =
        "{\"collection\":\"docs\",\"sampler\":\"RANDOM\",\"nTrials\":2,\"kForMetrics\":5,"
            + "\"querySampleSize\":4,\"mMin\":8,\"mMax\":16,\"efMin\":50,\"efMax\":100}";
    HttpResponse<String> submit = postJson("/api/optimize/studies", dto);
    assertThat(submit.statusCode()).isEqualTo(202);
    String studyId = JSON.readTree(submit.body()).get("studyId").asText();
    waitFor(
        () -> {
          try {
            HttpResponse<String> r = get("/api/optimize/studies/" + studyId + "/trials");
            return r.statusCode() == 200
                && "COMPLETED".equals(JSON.readTree(r.body()).get("state").asText());
          } catch (Exception e) {
            return false;
          }
        },
        Duration.ofSeconds(45));
    HttpResponse<String> res =
        postJson(
            "/collections/docs/optimize/apply/" + studyId + "/no-such-trial-id?confirm=true", "");
    assertThat(res.statusCode()).isEqualTo(404);
    assertThat(res.body()).contains("no-such-trial-id");
  }
}
