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
package com.integrallis.vectors.studio.sidecart.sources;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import com.integrallis.vectors.studio.sidecart.TextSearchHit;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class D1SidecartSourceTest {

  private D1HttpClient newClient(StubHttpClient stub) {
    return new D1HttpClient(stub, "acct", "db", "tok");
  }

  private D1SidecartSource newSource(D1HttpClient http) {
    return new D1SidecartSource(http, "docs", "doc_id", "content", "payload", "mime_type");
  }

  @Test
  void getDecodesD1ResponseAndBuildsRecord() {
    StubHttpClient stub = new StubHttpClient();
    String body = Base64.getEncoder().encodeToString(new byte[] {1, 2, 3});
    stub.enqueue(
        "{\"success\":true,\"result\":[{\"results\":[{"
            + "\"doc_id\":\"a\",\"content\":\"hello\",\"payload\":\""
            + body
            + "\",\"mime_type\":\"text/plain\"}]}]}");
    Optional<SidecartRecord> rec = newSource(newClient(stub)).get("a");
    assertThat(rec).isPresent();
    assertThat(rec.get().text()).isEqualTo("hello");
    assertThat(rec.get().blob()).isEqualTo(new byte[] {1, 2, 3});
    assertThat(rec.get().mime()).isEqualTo("text/plain");
  }

  @Test
  void getAllSendsBatchedInClauseInSqlAndParams() {
    StubHttpClient stub = new StubHttpClient();
    stub.enqueue(
        "{\"success\":true,\"result\":[{\"results\":["
            + "{\"doc_id\":\"a\",\"content\":\"alpha\",\"payload\":null,\"mime_type\":null},"
            + "{\"doc_id\":\"c\",\"content\":\"charlie\",\"payload\":null,\"mime_type\":null}"
            + "]}]}");
    Map<String, SidecartRecord> got = newSource(newClient(stub)).getAll(List.of("a", "b", "c"));
    assertThat(got).containsOnlyKeys("a", "c");
    assertThat(got.get("a").text()).isEqualTo("alpha");
    assertThat(stub.bodies).hasSize(1);
    String body = stub.bodies.get(0);
    assertThat(body).contains("WHERE doc_id IN (?, ?, ?)");
    assertThat(body).contains("\"a\"").contains("\"b\"").contains("\"c\"");
  }

  @Test
  void textSearchUsesFts5MatchAndInvertsBm25Sign() {
    StubHttpClient stub = new StubHttpClient();
    stub.enqueue(
        "{\"success\":true,\"result\":[{\"results\":["
            + "{\"id\":\"food\",\"bm25\":-1.5},"
            + "{\"id\":\"car\",\"bm25\":-0.3}"
            + "]}]}");
    List<TextSearchHit> hits = newSource(newClient(stub)).textSearch("sushi", 5);
    assertThat(hits).hasSize(2);
    assertThat(hits.get(0).id()).isEqualTo("food");
    // The source negates bm25 so larger == better.
    assertThat(hits.get(0).score()).isEqualTo(1.5);
    String body = stub.bodies.get(0);
    assertThat(body).contains("docs_fts MATCH ?");
    assertThat(body).contains("bm25(docs_fts)");
    assertThat(body).contains("\"sushi\"");
    assertThat(body).contains("\"params\":[\"sushi\",5]");
  }

  @Test
  void buildRequestBodyShape() {
    String body = D1SidecartSource.buildRequestBody("SELECT 1", List.of("x", 7L));
    assertThat(body).isEqualTo("{\"sql\":\"SELECT 1\",\"params\":[\"x\",7]}");
  }

  @Test
  void emptyIdSkipsTheNetwork() {
    StubHttpClient stub = new StubHttpClient();
    Optional<SidecartRecord> rec = newSource(newClient(stub)).get("");
    assertThat(rec).isEmpty();
    assertThat(stub.requests).isEmpty();
  }
}
