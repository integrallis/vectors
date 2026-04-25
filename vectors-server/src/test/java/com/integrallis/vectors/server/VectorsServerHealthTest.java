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

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import org.junit.jupiter.api.Test;

class VectorsServerHealthTest {

  @Test
  void healthEndpointReturnsUp() throws Exception {
    try (VectorsServer.ServerHandle handle = VectorsServer.start(ServerConfig.forTesting())) {
      HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
      HttpRequest request =
          HttpRequest.newBuilder()
              .uri(URI.create("http://localhost:" + handle.port() + "/v1/health"))
              .timeout(Duration.ofSeconds(5))
              .GET()
              .build();
      HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

      assertThat(response.statusCode()).isEqualTo(200);
      assertThat(response.headers().firstValue("content-type"))
          .hasValueSatisfying(v -> assertThat(v).startsWith("application/json"));
      assertThat(response.body()).contains("\"status\":\"UP\"").contains("\"collections\":0");
    }
  }

  @Test
  void ephemeralPortIsAssigned() {
    try (VectorsServer.ServerHandle handle = VectorsServer.start(ServerConfig.forTesting())) {
      assertThat(handle.port()).isPositive();
    }
  }
}
