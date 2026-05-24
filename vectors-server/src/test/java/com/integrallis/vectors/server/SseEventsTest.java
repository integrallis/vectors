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

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class SseEventsTest {

  @Test
  void sseEmitsHelloAndEpochEvents() throws Exception {
    try (VectorsServer.ServerHandle handle = VectorsServer.start(ServerConfig.forTesting())) {
      // 30s connect timeout (rather than the original 5s) to tolerate CPU starvation when the
      // root `./gradlew clean build test` runs many test JVMs in parallel — server startup +
      // first request can take more than 5s of wall-clock under that load even though the test is
      // sub-second in isolation.
      HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
      int port = handle.port();

      // Pre-create one collection so we can fire a commit against it later.
      postJson(
          client,
          port,
          "/v1/collections",
          "{\"name\":\"stream\",\"dimension\":4,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}");

      // Open the SSE stream.
      HttpRequest sseReq =
          HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/v1/events"))
              .timeout(Duration.ofSeconds(30))
              .header("accept", "text/event-stream")
              .GET()
              .build();
      HttpResponse<InputStream> sseRes = client.send(sseReq, BodyHandlers.ofInputStream());
      assertThat(sseRes.statusCode()).isEqualTo(200);
      assertThat(sseRes.headers().firstValue("content-type"))
          .hasValueSatisfying(v -> assertThat(v).startsWith("text/event-stream"));

      // Read lines on a dedicated virtual thread; collect until we have seen an epoch event.
      List<String> lines = new ArrayList<>();
      CountDownLatch gotEpoch = new CountDownLatch(1);
      Thread reader =
          Thread.startVirtualThread(
              () -> {
                try (BufferedReader r =
                    new BufferedReader(
                        new InputStreamReader(sseRes.body(), StandardCharsets.UTF_8))) {
                  String ln;
                  while ((ln = r.readLine()) != null) {
                    synchronized (lines) {
                      lines.add(ln);
                    }
                    if (ln.contains("\"epoch\":1")) {
                      gotEpoch.countDown();
                      break;
                    }
                  }
                } catch (Exception ignored) {
                  // connection closed by server teardown
                }
              });

      // Give the hello event a moment, then commit one document to fire an epoch event.
      Thread.sleep(150);
      postJson(
          client,
          port,
          "/v1/collections/stream/documents",
          "{\"documents\":[{\"id\":\"a\",\"vector\":[1,0,0,0]}]}");

      assertThat(gotEpoch.await(5, TimeUnit.SECONDS))
          .as("expected SSE to emit an epoch event within 5s")
          .isTrue();
      reader.join(Duration.ofSeconds(2));

      synchronized (lines) {
        String joined = String.join("\n", lines);
        assertThat(joined).contains("event:hello");
        assertThat(joined).contains("event:epoch");
        assertThat(joined).contains("\"name\":\"stream\"");
        assertThat(joined).contains("\"epoch\":1");
      }
    }
  }

  private static HttpResponse<String> postJson(
      HttpClient client, int port, String path, String body) throws Exception {
    return client.send(
        HttpRequest.newBuilder(URI.create("http://localhost:" + port + path))
            .timeout(Duration.ofSeconds(30))
            .header("content-type", "application/json")
            .POST(BodyPublishers.ofString(body))
            .build(),
        BodyHandlers.ofString());
  }
}
