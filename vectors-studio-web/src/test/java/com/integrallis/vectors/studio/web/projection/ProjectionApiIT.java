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
package com.integrallis.vectors.studio.web.projection;

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
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class ProjectionApiIT {

  private static final ObjectMapper JSON = new ObjectMapper();
  private static final int DIM = 8;

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
    Random rng = new Random(7);
    for (int i = 0; i < 60; i++) {
      float[] v = new float[DIM];
      for (int j = 0; j < DIM; j++) v[j] = (float) rng.nextGaussian();
      c.add(new Document("doc-" + i, v, null, Map.of()));
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

  private URI uri(String p) {
    return URI.create("http://localhost:" + handle.port() + p);
  }

  @Test
  void submitsPcaAndStreamsTerminalEvent() throws Exception {
    String body =
        "{\"collection\":\"docs\",\"algorithm\":\"PCA\",\"dimensions\":2,\"sampleSize\":0}";
    HttpResponse<String> submit =
        client.send(
            HttpRequest.newBuilder(uri("/api/projections"))
                .timeout(Duration.ofSeconds(10))
                .header("content-type", "application/json")
                .POST(BodyPublishers.ofString(body))
                .build(),
            BodyHandlers.ofString());
    assertThat(submit.statusCode()).isEqualTo(202);
    JsonNode submitBody = JSON.readTree(submit.body());
    String jobId = submitBody.get("jobId").asText();
    assertThat(submitBody.get("n").asInt()).isEqualTo(60);

    // Allow the projection job (PCA on 60 vectors) to settle before subscribing — the SSE
    // handler replays the terminal state for already-done jobs.
    Thread.sleep(200);

    HttpResponse<String> events =
        client.send(
            HttpRequest.newBuilder(uri("/api/projections/" + jobId + "/events"))
                .timeout(Duration.ofSeconds(15))
                .header("Accept", "text/event-stream")
                .GET()
                .build(),
            BodyHandlers.ofString());
    assertThat(events.statusCode()).isEqualTo(200);
    assertThat(events.body()).contains("\"jobId\":\"" + jobId + "\"");
    assertThat(events.body()).contains("\"result\":");
    assertThat(events.body()).contains("\"algorithm\":\"PCA\"");
  }

  @Test
  void rejectsUnknownCollection() throws Exception {
    String body =
        "{\"collection\":\"missing\",\"algorithm\":\"PCA\",\"dimensions\":2,\"sampleSize\":0}";
    HttpResponse<String> r =
        client.send(
            HttpRequest.newBuilder(uri("/api/projections"))
                .timeout(Duration.ofSeconds(5))
                .header("content-type", "application/json")
                .POST(BodyPublishers.ofString(body))
                .build(),
            BodyHandlers.ofString());
    assertThat(r.statusCode()).isEqualTo(400);
  }
}
