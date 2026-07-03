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
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BulkRoutesTest {

  private static final ObjectMapper JSON = ObjectMapperHolder.shared();
  private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(5);
  private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(10);

  private VectorsServer.ServerHandle handle;
  private HttpClient client;

  @BeforeEach
  void setUp() throws Exception {
    handle = VectorsServer.start(ServerConfig.forTesting());
    client = HttpClient.newBuilder().connectTimeout(CONNECT_TIMEOUT).build();
    postJson(
        "/v1/collections",
        "{\"name\":\"bulk\",\"dimension\":4,\"metric\":\"COSINE\",\"indexType\":\"FLAT\"}");
    postJson(
        "/v1/collections/bulk/documents",
        "{\"documents\":["
            + "{\"id\":\"a\",\"vector\":[1.0,0.0,0.0,0.0],\"metadata\":{\"cat\":\"x\"}},"
            + "{\"id\":\"b\",\"vector\":[0.0,1.0,0.0,0.0],\"metadata\":{\"cat\":\"y\"}},"
            + "{\"id\":\"c\",\"vector\":[0.0,0.0,1.0,0.0],\"metadata\":{\"cat\":\"x\"}},"
            + "{\"id\":\"d\",\"vector\":[0.0,0.0,0.0,1.0],\"metadata\":{\"cat\":\"y\"}}]}");
  }

  @AfterEach
  void tearDown() {
    if (handle != null) handle.close();
  }

  private URI uri(String path) {
    return URI.create("http://localhost:" + handle.port() + path);
  }

  private HttpResponse<String> get(String path) throws Exception {
    return client.send(
        HttpRequest.newBuilder(uri(path)).timeout(REQUEST_TIMEOUT).GET().build(),
        BodyHandlers.ofString());
  }

  private HttpResponse<String> postJson(String path, String body) throws Exception {
    return client.send(
        HttpRequest.newBuilder(uri(path))
            .timeout(REQUEST_TIMEOUT)
            .header("content-type", "application/json")
            .POST(BodyPublishers.ofString(body))
            .build(),
        BodyHandlers.ofString());
  }

  @Test
  void documentsPaginatesAndReportsTotal() throws Exception {
    HttpResponse<String> r = get("/v1/collections/bulk/documents?offset=0&limit=2");
    assertThat(r.statusCode()).isEqualTo(200);
    JsonNode body = JSON.readTree(r.body());
    assertThat(body.get("total").asInt()).isEqualTo(4);
    assertThat(body.get("items").size()).isEqualTo(2);
    assertThat(body.get("items").get(0).hasNonNull("vector")).isFalse();
  }

  @Test
  void documentsIncludesVectorsWhenAsked() throws Exception {
    JsonNode body =
        JSON.readTree(get("/v1/collections/bulk/documents?limit=10&includeVectors=true").body());
    assertThat(body.get("items").get(0).get("vector").size()).isEqualTo(4);
  }

  @Test
  void documentsRejectsNegativeOffset() throws Exception {
    assertThat(get("/v1/collections/bulk/documents?offset=-1&limit=5").statusCode()).isEqualTo(400);
  }

  @Test
  void documentsRejectsTooLargeLimit() throws Exception {
    assertThat(get("/v1/collections/bulk/documents?limit=99999").statusCode()).isEqualTo(400);
  }

  @Test
  void documentsForUnknownCollectionReturns404() throws Exception {
    assertThat(get("/v1/collections/missing/documents").statusCode()).isEqualTo(404);
  }

  @Test
  void vectorsBatchReturnsRequestedVectorsAndReportsMissing() throws Exception {
    HttpResponse<String> r =
        postJson("/v1/collections/bulk/vectors-batch", "{\"ids\":[\"a\",\"b\",\"zzz\"]}");
    assertThat(r.statusCode()).isEqualTo(200);
    JsonNode body = JSON.readTree(r.body());
    assertThat(body.get("vectors").size()).isEqualTo(2);
    assertThat(body.get("missing").size()).isEqualTo(1);
    assertThat(body.get("missing").get(0).asText()).isEqualTo("zzz");
  }

  @Test
  void sampleHonoursCap() throws Exception {
    JsonNode body = JSON.readTree(get("/v1/collections/bulk/sample?n=3").body());
    assertThat(body.get("ids").size()).isEqualTo(3);
    assertThat(body.get("vectors").size()).isEqualTo(3);
    assertThat(body.hasNonNull("metadata")).isFalse();
  }

  @Test
  void sampleIncludesMetadataWhenAsked() throws Exception {
    JsonNode body =
        JSON.readTree(get("/v1/collections/bulk/sample?n=2&includeMetadata=true").body());
    assertThat(body.get("metadata").size()).isEqualTo(2);
    assertThat(body.get("metadata").get(0).has("cat")).isTrue();
  }

  @Test
  void sampleRejectsNonPositiveN() throws Exception {
    assertThat(get("/v1/collections/bulk/sample?n=0").statusCode()).isEqualTo(400);
  }

  /**
   * Pins that the malformed-body 400 returns a generic problem-detail rather than echoing the raw
   * Jackson parser message back to the client. Audit finding: server routes that forwarded {@code
   * e.getMessage()} could leak parser/exception internals.
   */
  @Test
  void vectorsBatchMalformedBodyReturnsGenericProblemNoJacksonInternals() throws Exception {
    HttpResponse<String> r = postJson("/v1/collections/bulk/vectors-batch", "{not json");
    assertThat(r.statusCode()).isEqualTo(400);
    assertThat(r.headers().firstValue("content-type"))
        .hasValueSatisfying(v -> assertThat(v).startsWith("application/problem+json"));
    String body = r.body();
    assertThat(body)
        .as("must not leak Jackson internals")
        .doesNotContainIgnoringCase("unexpected character")
        .doesNotContainIgnoringCase("JsonParseException")
        .doesNotContainIgnoringCase("JsonMappingException")
        .doesNotContain("at [Source:")
        .doesNotContain("line:");
    assertThat(body).containsIgnoringCase("malformed").containsIgnoringCase("json");
  }
}
