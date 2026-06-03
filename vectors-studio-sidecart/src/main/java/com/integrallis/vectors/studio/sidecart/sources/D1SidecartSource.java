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
import com.integrallis.vectors.studio.sidecart.TextSearchHit;
import java.net.http.HttpClient;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * Sidecart source backed by a Cloudflare D1 database, queried over the public {@code
 * /accounts/{id}/d1/database/{db}/query} HTTP endpoint with a Bearer API token.
 *
 * <p>{@link #get(String)} issues one parameterised SELECT; {@link #getAll(Collection)} batches into
 * a single {@code WHERE id IN (?, ?, …)} round-trip so paging in Studio is one HTTP call, not N.
 * {@link #textSearch(String, int)} runs SQLite FTS5 {@code MATCH} against the {@code docs_fts}
 * virtual table created by {@link D1SidecartWriter#ensureSchema()} and returns BM25 scores
 * re-mapped so larger means better.
 */
public final class D1SidecartSource implements SidecartSource {

  private static final Pattern IDENT_PATTERN = Pattern.compile("[A-Za-z_][A-Za-z0-9_]{0,63}");
  static final int MAX_D1_QUERY_PARAMS = 900;

  private final D1HttpClient http;
  private final String table;
  private final String idColumn;
  private final String textColumn;
  private final String blobColumn;
  private final String mimeColumn;
  private final boolean hasText;
  private final boolean hasBlob;
  private final boolean hasMime;

  public D1SidecartSource(
      String accountId,
      String databaseId,
      String apiToken,
      String table,
      String idColumn,
      String textColumn,
      String blobColumn,
      String mimeColumn) {
    this(
        new D1HttpClient(
            HttpClient.newHttpClient(),
            Objects.requireNonNull(accountId, "accountId"),
            Objects.requireNonNull(databaseId, "databaseId"),
            Objects.requireNonNull(apiToken, "apiToken")),
        table,
        idColumn,
        textColumn,
        blobColumn,
        mimeColumn);
  }

  /** Test seam: inject a pre-built {@link D1HttpClient} (e.g. one wrapping a stub HttpClient). */
  D1SidecartSource(
      D1HttpClient http,
      String table,
      String idColumn,
      String textColumn,
      String blobColumn,
      String mimeColumn) {
    this.http = Objects.requireNonNull(http, "http");
    this.table = require(table, "table");
    this.idColumn = require(idColumn, "idColumn");
    if (textColumn != null) require(textColumn, "textColumn");
    if (blobColumn != null) require(blobColumn, "blobColumn");
    if (mimeColumn != null) require(mimeColumn, "mimeColumn");
    if (textColumn == null && blobColumn == null) {
      throw new IllegalArgumentException("at least one of textColumn / blobColumn must be set");
    }
    this.textColumn = textColumn;
    this.blobColumn = blobColumn;
    this.mimeColumn = mimeColumn;
    this.hasText = textColumn != null;
    this.hasBlob = blobColumn != null;
    this.hasMime = mimeColumn != null;
  }

  @Override
  public Optional<SidecartRecord> get(String id) {
    if (id == null || id.isEmpty()) return Optional.empty();
    Map<String, SidecartRecord> all = getAll(List.of(id));
    return Optional.ofNullable(all.get(id));
  }

  @Override
  public Map<String, SidecartRecord> getAll(Collection<String> ids) {
    if (ids == null || ids.isEmpty()) return Map.of();
    List<String> list = new ArrayList<>(ids);
    Map<String, SidecartRecord> out = new HashMap<>(list.size());
    for (int offset = 0; offset < list.size(); offset += MAX_D1_QUERY_PARAMS) {
      int end = Math.min(offset + MAX_D1_QUERY_PARAMS, list.size());
      fetchChunk(list.subList(offset, end), out);
    }
    return out;
  }

  private void fetchChunk(List<String> ids, Map<String, SidecartRecord> out) {
    StringBuilder cols = new StringBuilder();
    cols.append(idColumn);
    if (hasText) cols.append(", ").append(textColumn);
    if (hasBlob) cols.append(", ").append(blobColumn);
    if (hasMime) cols.append(", ").append(mimeColumn);
    StringBuilder placeholders = new StringBuilder();
    for (int i = 0; i < ids.size(); i++) placeholders.append(i == 0 ? "?" : ", ?");
    String sql =
        "SELECT " + cols + " FROM " + table + " WHERE " + idColumn + " IN (" + placeholders + ")";
    List<Map<String, Object>> rows = http.queryRows(sql, ids);
    for (Map<String, Object> row : rows) {
      Object idObj = row.get(idColumn);
      if (idObj == null) continue;
      String key = String.valueOf(idObj);
      String text = hasText ? asString(row.get(textColumn)) : null;
      byte[] blob = hasBlob ? asBytes(row.get(blobColumn)) : null;
      String mime = hasMime ? asString(row.get(mimeColumn)) : null;
      out.put(key, new SidecartRecord(text, blob, mime));
    }
  }

  @Override
  public List<TextSearchHit> textSearch(String query, int k) {
    if (query == null || query.isBlank() || k <= 0) return List.of();
    if (!hasText) return List.of();
    // External-content FTS5 layout: docs_fts.rowid -> docs.rowid (see D1SidecartWriter).
    // bm25 is "smaller is better" in SQLite; we negate so larger = more relevant.
    String sql =
        "SELECT d."
            + idColumn
            + " AS id, bm25("
            + table
            + "_fts) AS bm25 FROM "
            + table
            + "_fts "
            + "JOIN "
            + table
            + " d ON d.rowid = "
            + table
            + "_fts.rowid "
            + "WHERE "
            + table
            + "_fts MATCH ? ORDER BY bm25 LIMIT ?";
    List<Map<String, Object>> rows = http.queryRows(sql, List.of(query, (long) k));
    List<TextSearchHit> hits = new ArrayList<>(rows.size());
    for (Map<String, Object> row : rows) {
      Object idVal = row.get("id");
      Object scoreVal = row.get("bm25");
      if (idVal == null || scoreVal == null) continue;
      double bm25 = ((Number) scoreVal).doubleValue();
      hits.add(new TextSearchHit(String.valueOf(idVal), -bm25));
    }
    return hits;
  }

  @Override
  public void close() {
    http.close();
  }

  // ─────────── helpers ───────────

  private static String require(String ident, String name) {
    if (ident == null || !IDENT_PATTERN.matcher(ident).matches()) {
      throw new IllegalArgumentException(name + " is not a safe SQL identifier: " + ident);
    }
    return ident;
  }

  private static String asString(Object v) {
    return v == null ? null : v.toString();
  }

  private static byte[] asBytes(Object v) {
    if (v == null) return null;
    // D1 returns BLOB columns as base64 strings on the wire.
    if (v instanceof String s) {
      try {
        return Base64.getDecoder().decode(s);
      } catch (IllegalArgumentException e) {
        throw new SidecartSourceException("invalid base64 blob from D1", e);
      }
    }
    if (v instanceof List<?> list) {
      // Some D1 envelopes encode small blobs as int[] arrays.
      byte[] out = new byte[list.size()];
      for (int i = 0; i < list.size(); i++) out[i] = ((Number) list.get(i)).byteValue();
      return out;
    }
    throw new SidecartSourceException("unexpected blob shape: " + v.getClass(), null);
  }

  /** Build the JSON request body for a parameterised SQL execution. */
  static String buildRequestBody(String sql, List<?> params) {
    Map<String, Object> body = new LinkedHashMap<>();
    body.put("sql", sql);
    body.put("params", params == null ? List.of() : params);
    return D1Json.encode(body);
  }
}
