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

import java.net.http.HttpClient;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Writes documents into a Cloudflare D1 sidecart table and maintains a SQLite FTS5 virtual table
 * over its text column. {@link #ensureSchema()} issues the main table, the {@code <table>_fts}
 * virtual table, and the three insert/delete/update sync triggers in a single multi-statement HTTP
 * call. {@link #putAll(Iterable)} chunks rows into batched HTTP round-trips so a 1 000-row seed
 * costs ~20 calls rather than 1 000.
 *
 * <p>{@link D1SidecartSource#textSearch(String, int)} reads the FTS5 index this writer maintains.
 */
public final class D1SidecartWriter implements AutoCloseable {

  private static final Pattern IDENT_PATTERN = Pattern.compile("[A-Za-z_][A-Za-z0-9_]{0,63}");
  private static final int BATCH_ROWS = 50;

  private final D1HttpClient http;
  private final String table;
  private final String idCol;
  private final String textCol;
  private final String blobCol;
  private final String mimeCol;

  public D1SidecartWriter(
      String accountId,
      String databaseId,
      String apiToken,
      String table,
      String idCol,
      String textCol,
      String blobCol,
      String mimeCol) {
    this(
        new D1HttpClient(
            HttpClient.newHttpClient(),
            Objects.requireNonNull(accountId, "accountId"),
            Objects.requireNonNull(databaseId, "databaseId"),
            Objects.requireNonNull(apiToken, "apiToken")),
        table,
        idCol,
        textCol,
        blobCol,
        mimeCol);
  }

  /** Test seam: inject a pre-built {@link D1HttpClient}. */
  D1SidecartWriter(
      D1HttpClient http,
      String table,
      String idCol,
      String textCol,
      String blobCol,
      String mimeCol) {
    this.http = Objects.requireNonNull(http, "http");
    this.table = require(table, "table");
    this.idCol = require(idCol, "idCol");
    this.textCol = require(textCol, "textCol");
    this.blobCol = blobCol == null ? null : require(blobCol, "blobCol");
    this.mimeCol = mimeCol == null ? null : require(mimeCol, "mimeCol");
  }

  /**
   * Creates the main table, the external-content FTS5 virtual table, and the three sync triggers.
   * All statements are {@code IF NOT EXISTS} so re-running is a no-op.
   */
  public void ensureSchema() {
    String fts = table + "_fts";
    StringBuilder cols = new StringBuilder();
    cols.append("rowid INTEGER PRIMARY KEY AUTOINCREMENT, ");
    cols.append(idCol).append(" TEXT UNIQUE NOT NULL, ");
    cols.append(textCol).append(" TEXT");
    if (blobCol != null) cols.append(", ").append(blobCol).append(" BLOB");
    if (mimeCol != null) cols.append(", ").append(mimeCol).append(" TEXT");
    List<Map<String, Object>> stmts = new ArrayList<>();
    stmts.add(stmt("CREATE TABLE IF NOT EXISTS " + table + " (" + cols + ")"));
    stmts.add(
        stmt(
            "CREATE VIRTUAL TABLE IF NOT EXISTS "
                + fts
                + " USING fts5("
                + textCol
                + ", content='"
                + table
                + "', content_rowid='rowid')"));
    stmts.add(
        stmt(
            "CREATE TRIGGER IF NOT EXISTS "
                + table
                + "_ai AFTER INSERT ON "
                + table
                + " BEGIN "
                + "INSERT INTO "
                + fts
                + "(rowid, "
                + textCol
                + ") VALUES (new.rowid, new."
                + textCol
                + "); END"));
    stmts.add(
        stmt(
            "CREATE TRIGGER IF NOT EXISTS "
                + table
                + "_ad AFTER DELETE ON "
                + table
                + " BEGIN "
                + "INSERT INTO "
                + fts
                + "("
                + fts
                + ", rowid, "
                + textCol
                + ") VALUES('delete', old.rowid, old."
                + textCol
                + "); END"));
    stmts.add(
        stmt(
            "CREATE TRIGGER IF NOT EXISTS "
                + table
                + "_au AFTER UPDATE ON "
                + table
                + " BEGIN "
                + "INSERT INTO "
                + fts
                + "("
                + fts
                + ", rowid, "
                + textCol
                + ") VALUES('delete', old.rowid, old."
                + textCol
                + "); "
                + "INSERT INTO "
                + fts
                + "(rowid, "
                + textCol
                + ") VALUES (new.rowid, new."
                + textCol
                + "); END"));
    http.executeBatch(stmts);
  }

  /** Upserts a single row. */
  public void put(String id, String text, byte[] blob, String mime) {
    putAll(List.of(new SidecartUpsert(id, text, blob, mime)));
  }

  /** Upserts each row, chunked into multi-statement HTTP batches of {@value #BATCH_ROWS}. */
  public void putAll(Iterable<SidecartUpsert> rows) {
    String sql = upsertSql();
    List<Map<String, Object>> batch = new ArrayList<>(BATCH_ROWS);
    for (SidecartUpsert row : rows) {
      List<Object> params = new ArrayList<>(4);
      params.add(row.id());
      params.add(row.text());
      if (blobCol != null)
        params.add(row.blob() == null ? null : Base64.getEncoder().encodeToString(row.blob()));
      if (mimeCol != null) params.add(row.mime());
      Map<String, Object> stmt = new LinkedHashMap<>();
      stmt.put("sql", sql);
      stmt.put("params", params);
      batch.add(stmt);
      if (batch.size() >= BATCH_ROWS) {
        http.executeBatch(batch);
        batch = new ArrayList<>(BATCH_ROWS);
      }
    }
    if (!batch.isEmpty()) http.executeBatch(batch);
  }

  @Override
  public void close() {
    http.close();
  }

  // ─────────── helpers ───────────

  private String upsertSql() {
    StringBuilder cols = new StringBuilder().append(idCol).append(", ").append(textCol);
    StringBuilder qs = new StringBuilder("?, ?");
    StringBuilder upd = new StringBuilder(textCol).append("=excluded.").append(textCol);
    if (blobCol != null) {
      cols.append(", ").append(blobCol);
      qs.append(", ?");
      upd.append(", ").append(blobCol).append("=excluded.").append(blobCol);
    }
    if (mimeCol != null) {
      cols.append(", ").append(mimeCol);
      qs.append(", ?");
      upd.append(", ").append(mimeCol).append("=excluded.").append(mimeCol);
    }
    return "INSERT INTO "
        + table
        + " ("
        + cols
        + ") VALUES ("
        + qs
        + ") "
        + "ON CONFLICT("
        + idCol
        + ") DO UPDATE SET "
        + upd;
  }

  private static Map<String, Object> stmt(String sql) {
    Map<String, Object> m = new LinkedHashMap<>();
    m.put("sql", sql);
    m.put("params", List.of());
    return m;
  }

  private static String require(String ident, String name) {
    if (ident == null || !IDENT_PATTERN.matcher(ident).matches()) {
      throw new IllegalArgumentException(name + " is not a safe SQL identifier: " + ident);
    }
    return ident;
  }
}
