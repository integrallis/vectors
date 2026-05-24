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
package com.integrallis.vectors.text.h2;

import com.integrallis.vectors.hybrid.text.TextIndexSpi;
import com.integrallis.vectors.hybrid.text.TextSearchOutcome;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * H2 embedded database implementation of {@link TextIndexSpi}.
 *
 * <p>Provides full-text search (via H2's built-in FTS), metadata storage (pipe-delimited {@code
 * key=value} pairs in a VARCHAR column — see {@link #encodeMetadata} and the documented encoding
 * caveats there), and blob storage (BLOB column for images).
 */
public final class H2TextIndex implements TextIndexSpi {

  private static final Logger LOG = LoggerFactory.getLogger(H2TextIndex.class);

  private final Connection connection;

  /**
   * Creates an in-memory H2 text index for the given collection name.
   *
   * @param collectionName used to namespace the database
   */
  public H2TextIndex(String collectionName) {
    this(collectionName, null);
  }

  /**
   * Creates an H2 text index for the given collection name. When {@code dataDir} is non-null, the
   * database is stored on disk at {@code dataDir/text-index.mv.db} and survives restarts. When
   * {@code dataDir} is {@code null}, the database is in-memory only.
   *
   * @param collectionName used to namespace the database
   * @param dataDir directory to store the database file, or {@code null} for in-memory
   */
  public H2TextIndex(String collectionName, Path dataDir) {
    Objects.requireNonNull(collectionName, "collectionName");
    try {
      String jdbcUrl;
      if (dataDir != null) {
        Path dbFile = dataDir.resolve("text-index");
        // Do NOT pre-delete H2's housekeeping files (.lock.db / .trace.db).
        //
        // .lock.db is exactly how H2 prevents two processes from opening the same database;
        // unconditionally removing it lets a second server (or a crashed-but-still-running first
        // one) silently stomp the first, causing data corruption. .trace.db carries diagnostics
        // from a prior unclean shutdown that we want to be able to read after the fact.
        //
        // If a genuinely stale lock survives a crash, the right response is an operator-visible
        // failure from DriverManager (which the surrounding try/catch surfaces), not a silent
        // recovery that hides the prior process.
        jdbcUrl = "jdbc:h2:file:" + dbFile.toAbsolutePath();
        LOG.debug("H2 text index persistent at {}", dbFile);
      } else {
        jdbcUrl = "jdbc:h2:mem:" + collectionName;
      }
      this.connection = DriverManager.getConnection(jdbcUrl);
      initSchema();
    } catch (SQLException e) {
      throw new IllegalStateException("failed to initialize H2 text index", e);
    }
  }

  private void initSchema() throws SQLException {
    try (Statement stmt = connection.createStatement()) {
      stmt.execute(
          """
          CREATE TABLE IF NOT EXISTS documents (
              id VARCHAR PRIMARY KEY,
              text_content CLOB,
              metadata VARCHAR,
              blob_data BLOB,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          )
          """);
      // H2 built-in full-text search (no Lucene dependency needed)
      stmt.execute("CREATE ALIAS IF NOT EXISTS FT_INIT FOR 'org.h2.fulltext.FullText.init'");
      stmt.execute("CALL FT_INIT()");
      try {
        stmt.execute("CALL FT_CREATE_INDEX('PUBLIC', 'DOCUMENTS', 'TEXT_CONTENT')");
      } catch (SQLException e) {
        // Index may already exist from a prior call
        if (!e.getMessage().contains("already")) {
          throw e;
        }
      }
    }
  }

  @Override
  public void index(List<TextDocument> documents) {
    Objects.requireNonNull(documents, "documents");
    try {
      for (TextDocument doc : documents) {
        // MERGE = upsert
        try (PreparedStatement ps =
            connection.prepareStatement(
                "MERGE INTO documents (id, text_content, metadata, blob_data) VALUES (?, ?, ?, ?)")) {
          ps.setString(1, doc.id());
          ps.setString(2, doc.text());
          ps.setString(3, encodeMetadata(doc.metadata()));
          ps.setBytes(4, doc.blob());
          ps.executeUpdate();
        }
      }
    } catch (SQLException e) {
      throw new IllegalStateException("failed to index documents", e);
    }
  }

  @Override
  public TextSearchOutcome search(String query, int k) {
    Objects.requireNonNull(query, "query");
    if (query.isBlank() || k <= 0) {
      return TextSearchOutcome.empty();
    }
    try {
      List<String> ids = new ArrayList<>();
      List<Float> scores = new ArrayList<>();

      // H2 FT_SEARCH_DATA returns TABLE(SCHEMA VARCHAR, TABLE VARCHAR, COLUMNS ARRAY, KEYS ARRAY)
      try (PreparedStatement ps =
          connection.prepareStatement("SELECT FT.KEYS FROM FT_SEARCH_DATA(?, 0, 0) FT LIMIT ?")) {
        ps.setString(1, query);
        ps.setInt(2, k);
        try (ResultSet rs = ps.executeQuery()) {
          float rank = 1.0f;
          while (rs.next()) {
            Object[] keys = (Object[]) rs.getArray(1).getArray();
            if (keys.length > 0) {
              String id = keys[0].toString();
              ids.add(id);
              // H2 FTS doesn't provide relevance scores; use reciprocal rank as proxy
              scores.add(1.0f / rank);
              rank++;
            }
          }
        }
      }

      return new TextSearchOutcome(ids.toArray(new String[0]), toFloatArray(scores));
    } catch (SQLException e) {
      LOG.warn("FTS query failed for '{}': {}", query, e.getMessage());
      return TextSearchOutcome.empty();
    }
  }

  @Override
  public Optional<StoredContent> get(String id) {
    Objects.requireNonNull(id, "id");
    try (PreparedStatement ps =
        connection.prepareStatement("SELECT text_content, metadata FROM documents WHERE id = ?")) {
      ps.setString(1, id);
      try (ResultSet rs = ps.executeQuery()) {
        if (rs.next()) {
          return Optional.of(
              new StoredContent(
                  id, rs.getString("text_content"), decodeMetadata(rs.getString("metadata"))));
        }
      }
    } catch (SQLException e) {
      throw new IllegalStateException("failed to get document " + id, e);
    }
    return Optional.empty();
  }

  @Override
  public Optional<byte[]> getBlob(String id) {
    Objects.requireNonNull(id, "id");
    try (PreparedStatement ps =
        connection.prepareStatement("SELECT blob_data FROM documents WHERE id = ?")) {
      ps.setString(1, id);
      try (ResultSet rs = ps.executeQuery()) {
        if (rs.next()) {
          byte[] blob = rs.getBytes("blob_data");
          return Optional.ofNullable(blob);
        }
      }
    } catch (SQLException e) {
      throw new IllegalStateException("failed to get blob for " + id, e);
    }
    return Optional.empty();
  }

  @Override
  public void remove(String id) {
    Objects.requireNonNull(id, "id");
    try (PreparedStatement ps = connection.prepareStatement("DELETE FROM documents WHERE id = ?")) {
      ps.setString(1, id);
      ps.executeUpdate();
    } catch (SQLException e) {
      throw new IllegalStateException("failed to remove document " + id, e);
    }
  }

  @Override
  public void clear() {
    try (Statement stmt = connection.createStatement()) {
      stmt.execute("DELETE FROM documents");
    } catch (SQLException e) {
      throw new IllegalStateException("failed to clear text index", e);
    }
  }

  @Override
  public int size() {
    try (Statement stmt = connection.createStatement();
        ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM documents")) {
      rs.next();
      return rs.getInt(1);
    } catch (SQLException e) {
      throw new IllegalStateException("failed to get text index size", e);
    }
  }

  @Override
  public void close() {
    try {
      connection.close();
    } catch (SQLException e) {
      LOG.warn("error closing H2 connection", e);
    }
  }

  @Override
  public void drop() {
    try (Statement stmt = connection.createStatement()) {
      stmt.execute("DROP ALL OBJECTS DELETE FILES");
    } catch (SQLException e) {
      LOG.warn("error dropping H2 database objects", e);
    } finally {
      close();
    }
  }

  // --- metadata encoding (simple key=value pairs, pipe-delimited) ---

  private static String encodeMetadata(Map<String, String> metadata) {
    if (metadata == null || metadata.isEmpty()) {
      return "";
    }
    StringBuilder sb = new StringBuilder();
    for (Map.Entry<String, String> entry : metadata.entrySet()) {
      if (!sb.isEmpty()) {
        sb.append('|');
      }
      sb.append(entry.getKey()).append('=').append(entry.getValue());
    }
    return sb.toString();
  }

  private static Map<String, String> decodeMetadata(String encoded) {
    Map<String, String> map = new HashMap<>();
    if (encoded == null || encoded.isEmpty()) {
      return map;
    }
    for (String pair : encoded.split("\\|")) {
      int eq = pair.indexOf('=');
      if (eq > 0) {
        map.put(pair.substring(0, eq), pair.substring(eq + 1));
      }
    }
    return map;
  }

  private static float[] toFloatArray(List<Float> list) {
    float[] result = new float[list.size()];
    for (int i = 0; i < list.size(); i++) {
      result[i] = list.get(i);
    }
    return result;
  }
}
