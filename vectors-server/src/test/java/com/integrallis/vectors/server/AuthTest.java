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
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import org.junit.jupiter.api.Test;

class AuthTest {

  private static final String CREATE_COLLECTION =
      """
      {"name":"secure","dimension":4,"metric":"COSINE","indexType":"FLAT"}
      """;

  private final HttpClient client =
      HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

  @Test
  void protectedRoutesAllowRequestsWhenAuthIsDisabled() throws Exception {
    try (VectorsServer.ServerHandle handle = VectorsServer.start(ServerConfig.forTesting())) {
      HttpResponse<String> response = get(handle, "/v1/collections", null);

      assertThat(response.statusCode()).isEqualTo(200);
    }
  }

  @Test
  void healthAndReadyRoutesStayPublicWhenAuthIsEnabled() throws Exception {
    try (VectorsServer.ServerHandle handle =
        VectorsServer.start(ServerConfig.forTesting().withApiKey("secret"))) {
      assertThat(get(handle, "/v1/health", null).statusCode()).isEqualTo(200);
      assertThat(get(handle, "/v1/readyz", null).statusCode()).isEqualTo(200);
    }
  }

  @Test
  void protectedRoutesRejectMissingOrWrongBearerToken() throws Exception {
    try (VectorsServer.ServerHandle handle =
        VectorsServer.start(ServerConfig.forTesting().withApiKey("secret"))) {
      HttpResponse<String> missing = post(handle, "/v1/collections", CREATE_COLLECTION, null);
      HttpResponse<String> wrong =
          post(handle, "/v1/collections", CREATE_COLLECTION, "Bearer wrong");

      assertThat(missing.statusCode()).isEqualTo(401);
      assertThat(wrong.statusCode()).isEqualTo(401);
      assertThat(missing.headers().firstValue("www-authenticate")).contains("Bearer");
    }
  }

  @Test
  void protectedRoutesAcceptMatchingBearerToken() throws Exception {
    try (VectorsServer.ServerHandle handle =
        VectorsServer.start(ServerConfig.forTesting().withApiKey("secret"))) {
      HttpResponse<String> response =
          post(handle, "/v1/collections", CREATE_COLLECTION, "Bearer secret");

      assertThat(response.statusCode()).isEqualTo(201);
    }
  }

  @Test
  void metricsRequiresBearerTokenWhenAuthIsEnabled() throws Exception {
    try (VectorsServer.ServerHandle handle =
        VectorsServer.start(ServerConfig.forTesting().withApiKey("secret"))) {
      assertThat(get(handle, "/v1/metrics", null).statusCode()).isEqualTo(401);
      assertThat(get(handle, "/v1/metrics", "Bearer secret").statusCode()).isEqualTo(200);
    }
  }

  private HttpResponse<String> get(VectorsServer.ServerHandle handle, String path, String auth)
      throws Exception {
    HttpRequest.Builder builder =
        HttpRequest.newBuilder(uri(handle, path)).timeout(Duration.ofSeconds(5)).GET();
    if (auth != null) {
      builder.header("authorization", auth);
    }
    return client.send(builder.build(), BodyHandlers.ofString());
  }

  private HttpResponse<String> post(
      VectorsServer.ServerHandle handle, String path, String body, String auth) throws Exception {
    HttpRequest.Builder builder =
        HttpRequest.newBuilder(uri(handle, path))
            .timeout(Duration.ofSeconds(10))
            .header("content-type", "application/json")
            .POST(BodyPublishers.ofString(body));
    if (auth != null) {
      builder.header("authorization", auth);
    }
    return client.send(builder.build(), BodyHandlers.ofString());
  }

  private static URI uri(VectorsServer.ServerHandle handle, String path) {
    return URI.create("http://localhost:" + handle.port() + path);
  }
}
