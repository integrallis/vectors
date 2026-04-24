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
