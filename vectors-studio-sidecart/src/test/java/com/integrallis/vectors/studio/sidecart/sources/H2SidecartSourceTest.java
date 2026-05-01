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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class H2SidecartSourceTest {

  private static final String JDBC = "jdbc:h2:mem:test_sidecart;DB_CLOSE_DELAY=-1;MODE=LEGACY";

  @BeforeEach
  void seed() throws Exception {
    try (Connection c = DriverManager.getConnection(JDBC, "sa", "");
        Statement s = c.createStatement()) {
      s.execute("DROP TABLE IF EXISTS docs");
      s.execute(
          "CREATE TABLE docs ("
              + "  doc_id VARCHAR(64) PRIMARY KEY,"
              + "  content CLOB,"
              + "  payload VARBINARY(1024),"
              + "  mime_type VARCHAR(64))");
      s.execute(
          "INSERT INTO docs(doc_id, content, payload, mime_type) "
              + "VALUES ('a', 'hello', X'010203', 'text/plain')");
      s.execute(
          "INSERT INTO docs(doc_id, content, payload, mime_type) "
              + "VALUES ('img', NULL, X'89504E47', 'image/png')");
    }
  }

  @Test
  void readsAllConfiguredColumns() {
    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", "content", "payload", "mime_type");
    Optional<SidecartRecord> rec = src.get("a");
    assertThat(rec).isPresent();
    assertThat(rec.get().text()).isEqualTo("hello");
    assertThat(rec.get().blob()).isEqualTo(new byte[] {1, 2, 3});
    assertThat(rec.get().mime()).isEqualTo("text/plain");
  }

  @Test
  void readsBlobOnlyWhenTextColumnNotConfigured() {
    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", null, "payload", "mime_type");
    Optional<SidecartRecord> rec = src.get("img");
    assertThat(rec).isPresent();
    assertThat(rec.get().text()).isNull();
    assertThat(rec.get().blob()).isEqualTo(new byte[] {(byte) 0x89, 0x50, 0x4E, 0x47});
    assertThat(rec.get().mime()).isEqualTo("image/png");
  }

  @Test
  void readsTextOnlyWhenBlobColumnNotConfigured() {
    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    Optional<SidecartRecord> rec = src.get("a");
    assertThat(rec).isPresent();
    assertThat(rec.get().text()).isEqualTo("hello");
    assertThat(rec.get().blob()).isNull();
    assertThat(rec.get().mime()).isNull();
  }

  @Test
  void emptyForUnknownId() {
    H2SidecartSource src =
        new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", "content", null, null);
    assertThat(src.get("missing")).isEmpty();
  }

  @Test
  void rejectsUnsafeIdentifiers() {
    assertThatThrownBy(
            () ->
                new H2SidecartSource(
                    JDBC, "sa", "", "docs; DROP TABLE docs", "doc_id", "content", null, null))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsAllNullProjection() {
    assertThatThrownBy(
            () -> new H2SidecartSource(JDBC, "sa", "", "docs", "doc_id", null, null, null))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
