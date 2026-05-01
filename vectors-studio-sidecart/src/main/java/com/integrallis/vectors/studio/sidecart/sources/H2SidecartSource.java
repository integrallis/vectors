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

import com.integrallis.vectors.studio.sidecart.SidecartRecord;
import com.integrallis.vectors.studio.sidecart.SidecartSource;
import com.integrallis.vectors.studio.sidecart.SidecartSourceException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * Sidecart source backed by an H2 (or any JDBC-compatible) database. A single SELECT against the
 * configured table reads {@code idColumn}, the optional {@code textColumn}, the optional {@code
 * blobColumn}, and the optional {@code mimeColumn} for the requested id.
 *
 * <p>The schema, table, and column identifiers are validated against {@code IDENT_PATTERN} before
 * being interpolated into the SQL — {@link PreparedStatement} only parameterises values, not
 * identifiers — so callers <b>must</b> supply identifiers from a trusted configuration file, never
 * from end-user input.
 */
public final class H2SidecartSource implements SidecartSource {

  private static final Pattern IDENT_PATTERN = Pattern.compile("[A-Za-z_][A-Za-z0-9_]{0,63}");

  private final String jdbcUrl;
  private final String user;
  private final String password;
  private final String selectSql;
  private final boolean hasText;
  private final boolean hasBlob;
  private final boolean hasMime;

  public H2SidecartSource(
      String jdbcUrl,
      String user,
      String password,
      String table,
      String idColumn,
      String textColumn,
      String blobColumn,
      String mimeColumn) {
    this.jdbcUrl = Objects.requireNonNull(jdbcUrl, "jdbcUrl");
    this.user = user == null ? "" : user;
    this.password = password == null ? "" : password;
    require(table, "table");
    require(idColumn, "idColumn");
    if (textColumn != null) require(textColumn, "textColumn");
    if (blobColumn != null) require(blobColumn, "blobColumn");
    if (mimeColumn != null) require(mimeColumn, "mimeColumn");
    if (textColumn == null && blobColumn == null) {
      throw new IllegalArgumentException("at least one of textColumn / blobColumn must be set");
    }
    this.hasText = textColumn != null;
    this.hasBlob = blobColumn != null;
    this.hasMime = mimeColumn != null;
    StringBuilder cols = new StringBuilder();
    if (hasText) cols.append(textColumn);
    if (hasBlob) {
      if (cols.length() > 0) cols.append(", ");
      cols.append(blobColumn);
    }
    if (hasMime) cols.append(", ").append(mimeColumn);
    this.selectSql = "SELECT " + cols + " FROM " + table + " WHERE " + idColumn + " = ?";
  }

  @Override
  public Optional<SidecartRecord> get(String id) {
    if (id == null || id.isEmpty()) return Optional.empty();
    try (Connection conn = DriverManager.getConnection(jdbcUrl, user, password);
        PreparedStatement ps = conn.prepareStatement(selectSql)) {
      ps.setString(1, id);
      try (ResultSet rs = ps.executeQuery()) {
        if (!rs.next()) return Optional.empty();
        int idx = 1;
        String text = hasText ? rs.getString(idx++) : null;
        byte[] blob = hasBlob ? rs.getBytes(idx++) : null;
        String mime = hasMime ? rs.getString(idx) : null;
        return Optional.of(new SidecartRecord(text, blob, mime));
      }
    } catch (SQLException e) {
      throw new SidecartSourceException("H2 sidecart lookup failed for id=" + id, e);
    }
  }

  private static void require(String ident, String name) {
    if (ident == null || !IDENT_PATTERN.matcher(ident).matches()) {
      throw new IllegalArgumentException(name + " is not a safe SQL identifier: " + ident);
    }
  }
}
