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
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class H2SidecartWriterTest {

  private static final String JDBC = "jdbc:h2:mem:writer_test;DB_CLOSE_DELAY=-1;MODE=LEGACY";

  @BeforeEach
  void resetSchema() throws Exception {
    try (var c = java.sql.DriverManager.getConnection(JDBC, "sa", "");
        var s = c.createStatement()) {
      s.execute("DROP ALL OBJECTS");
    }
  }

  @Test
  void roundTripWriterToSourceEquality() {
    H2SidecartWriter w =
        new H2SidecartWriter(JDBC, "sa", "", "docs", "doc_id", "content", "payload", "mime_type");
    w.ensureSchema();
    w.put("a", "hello world", new byte[] {1, 2, 3}, "text/plain");
    w.put("b", "another doc", null, "text/plain");

    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", "content", "payload", "mime_type");
    SidecartRecord ra = src.get("a").orElseThrow();
    assertThat(ra.text()).isEqualTo("hello world");
    assertThat(ra.blob()).isEqualTo(new byte[] {1, 2, 3});
    assertThat(ra.mime()).isEqualTo("text/plain");
    SidecartRecord rb = src.get("b").orElseThrow();
    assertThat(rb.text()).isEqualTo("another doc");
    assertThat(rb.blob()).isNull();
  }

  @Test
  void getAllReturnsBatchedRowsWithMissingIdsAbsent() {
    H2SidecartWriter w =
        new H2SidecartWriter(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    w.ensureSchema();
    w.putAll(
        List.of(
            new SidecartUpsert("a", "alpha", null, null),
            new SidecartUpsert("b", "bravo", null, null),
            new SidecartUpsert("c", "charlie", null, null)));

    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    Map<String, SidecartRecord> got = src.getAll(List.of("a", "missing", "c"));
    assertThat(got).containsOnlyKeys("a", "c");
    assertThat(got.get("a").text()).isEqualTo("alpha");
    assertThat(got.get("c").text()).isEqualTo("charlie");
  }

  @Test
  void textSearchReturnsMatchingIdsByToken() {
    H2SidecartWriter w =
        new H2SidecartWriter(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    w.ensureSchema();
    w.putAll(
        List.of(
            new SidecartUpsert("car", "the red car drove down the road", null, null),
            new SidecartUpsert("dog", "a happy dog ran in the park", null, null),
            new SidecartUpsert("food", "sushi rolls and ramen for dinner", null, null)));

    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    List<TextSearchHit> hits = src.textSearch("sushi", 5);
    assertThat(hits).extracting(TextSearchHit::id).contains("food");
  }

  @Test
  void putIsIdempotentAcrossCallsViaMerge() {
    H2SidecartWriter w =
        new H2SidecartWriter(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    w.ensureSchema();
    w.put("a", "v1", null, null);
    w.put("a", "v2", null, null);

    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    assertThat(src.get("a").orElseThrow().text()).isEqualTo("v2");
  }
}
