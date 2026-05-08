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

import com.integrallis.vectors.studio.sidecart.SidecartSourceException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Writes documents into an H2 sidecart table and maintains an H2 native fulltext index over its
 * text column. The schema is created on demand via {@link #ensureSchema()}; subsequent {@link #put}
 * / {@link #putAll} calls upsert rows using {@code MERGE INTO}, and the fulltext inverted index is
 * kept current automatically by H2's fulltext triggers.
 *
 * <p>{@link H2SidecartSource#textSearch(String, int)} reads the same index via {@code
 * FT_SEARCH_DATA}.
 */
public final class H2SidecartWriter implements AutoCloseable {

  private static final Pattern IDENT_PATTERN = Pattern.compile("[A-Za-z_][A-Za-z0-9_]{0,63}");

  private final String jdbcUrl;
  private final String user;
  private final String password;
  private final String table;
  private final String idCol;
  private final String textCol;
  private final String blobCol;
  private final String mimeCol;

  public H2SidecartWriter(
      String jdbcUrl,
      String user,
      String password,
      String table,
      String idCol,
      String textCol,
      String blobCol,
      String mimeCol) {
    this.jdbcUrl = Objects.requireNonNull(jdbcUrl, "jdbcUrl");
    this.user = user == null ? "" : user;
    this.password = password == null ? "" : password;
    this.table = require(table, "table");
    this.idCol = require(idCol, "idCol");
    this.textCol = require(textCol, "textCol");
    this.blobCol = blobCol == null ? null : require(blobCol, "blobCol");
    this.mimeCol = mimeCol == null ? null : require(mimeCol, "mimeCol");
  }

  /**
   * Creates the table (if absent) and initialises the H2 native fulltext index on the text column.
   * Idempotent.
   */
  public void ensureSchema() {
    StringBuilder cols = new StringBuilder();
    cols.append(idCol).append(" VARCHAR(255) PRIMARY KEY, ");
    cols.append(textCol).append(" CLOB");
    if (blobCol != null) cols.append(", ").append(blobCol).append(" VARBINARY(1048576)");
    if (mimeCol != null) cols.append(", ").append(mimeCol).append(" VARCHAR(127)");
    String createTable = "CREATE TABLE IF NOT EXISTS " + table + " (" + cols + ")";
    String tableUpper = table.toUpperCase(Locale.ROOT);
    String textColUpper = textCol.toUpperCase(Locale.ROOT);
    try (Connection conn = open();
        Statement st = conn.createStatement()) {
      st.execute(createTable);
      st.execute("CREATE ALIAS IF NOT EXISTS FT_INIT FOR \"org.h2.fulltext.FullText.init\"");
      st.execute("CALL FT_INIT()");
      // FT_CREATE_INDEX is a no-op when the index already exists.
      st.execute("CALL FT_CREATE_INDEX('PUBLIC', '" + tableUpper + "', '" + textColUpper + "')");
    } catch (SQLException e) {
      throw new SidecartSourceException("H2 sidecart ensureSchema failed", e);
    }
  }

  /** Upserts a single row. */
  public void put(String id, String text, byte[] blob, String mime) {
    putAll(java.util.List.of(new SidecartUpsert(id, text, blob, mime)));
  }

  /** Upserts each row via JDBC batch. */
  public void putAll(Iterable<SidecartUpsert> rows) {
    StringBuilder sql = new StringBuilder("MERGE INTO ").append(table).append(" (");
    sql.append(idCol).append(", ").append(textCol);
    if (blobCol != null) sql.append(", ").append(blobCol);
    if (mimeCol != null) sql.append(", ").append(mimeCol);
    sql.append(") KEY(").append(idCol).append(") VALUES (?, ?");
    if (blobCol != null) sql.append(", ?");
    if (mimeCol != null) sql.append(", ?");
    sql.append(")");
    try (Connection conn = open();
        PreparedStatement ps = conn.prepareStatement(sql.toString())) {
      int batched = 0;
      for (SidecartUpsert row : rows) {
        int idx = 1;
        ps.setString(idx++, row.id());
        ps.setString(idx++, row.text());
        if (blobCol != null) ps.setBytes(idx++, row.blob());
        if (mimeCol != null) ps.setString(idx, row.mime());
        ps.addBatch();
        batched++;
      }
      if (batched > 0) ps.executeBatch();
    } catch (SQLException e) {
      throw new SidecartSourceException("H2 sidecart upsert failed", e);
    }
  }

  @Override
  public void close() {
    // Stateless: each call opens a fresh connection. Nothing to release.
  }

  private Connection open() throws SQLException {
    return DriverManager.getConnection(jdbcUrl, user, password);
  }

  private static String require(String ident, String name) {
    if (ident == null || !IDENT_PATTERN.matcher(ident).matches()) {
      throw new IllegalArgumentException(name + " is not a safe SQL identifier: " + ident);
    }
    return ident;
  }
}
