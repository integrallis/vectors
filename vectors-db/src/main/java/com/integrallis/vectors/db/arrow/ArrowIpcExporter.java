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
package com.integrallis.vectors.db.arrow;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Exports a {@link VectorCollection} or a list of {@link Document}s to Apache Arrow IPC stream
 * format.
 *
 * <p><b>Arrow schema</b> produced by this exporter:
 *
 * <pre>
 * id            : Utf8       (not-null)
 * vector        : FixedSizeList(Float4, dim)  (not-null)
 * text          : Utf8       (nullable)
 * metadata_json : Utf8       (nullable) — JSON object with string/number/bool/string-array values
 * </pre>
 *
 * <p>The stream format is compatible with {@code pyarrow.ipc.open_stream()} for interoperability
 * with Python/NumPy/Pandas workflows. Use {@link ArrowIpcIngester} to read the stream back.
 */
public final class ArrowIpcExporter {

  /** Column name constants for the Arrow IPC schema. */
  public static final String COL_ID = "id";

  public static final String COL_VECTOR = "vector";
  public static final String COL_TEXT = "text";
  public static final String COL_METADATA_JSON = "metadata_json";

  /** Number of documents per Arrow record batch. */
  public static final int DEFAULT_BATCH_SIZE = 1_000;

  private ArrowIpcExporter() {}

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Exports all live documents from {@code collection} to an Arrow IPC stream on {@code out}. The
   * collection's {@link VectorCollection#documents()} is called once under the reader lock.
   *
   * @param collection the source collection (committed generation)
   * @param out the destination stream; not closed by this method
   * @throws IOException on serialization failure
   */
  public static void export(VectorCollection collection, OutputStream out) throws IOException {
    List<Document> docs = collection.documents();
    int dim = collection.config().dimension();
    export(docs, dim, out);
  }

  /**
   * Exports all live documents from {@code collection} to an Arrow IPC file at {@code file}.
   *
   * @param collection the source collection
   * @param file destination file path (created or overwritten)
   * @throws IOException on I/O or serialization failure
   */
  public static void export(VectorCollection collection, Path file) throws IOException {
    try (OutputStream out = Files.newOutputStream(file)) {
      export(collection, out);
    }
  }

  /**
   * Exports the given list of documents to an Arrow IPC stream on {@code out}.
   *
   * @param docs the documents to export
   * @param dimension vector dimension (must match each document's vector length)
   * @param out destination stream; not closed by this method
   * @throws IOException on serialization failure
   */
  public static void export(List<Document> docs, int dimension, OutputStream out)
      throws IOException {
    Schema schema = buildSchema(dimension);
    try (BufferAllocator allocator = new RootAllocator();
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
      writer.start();
      int total = docs.size();
      for (int batchStart = 0; batchStart < total || total == 0; batchStart += DEFAULT_BATCH_SIZE) {
        if (total == 0) break; // empty collection
        int batchEnd = Math.min(batchStart + DEFAULT_BATCH_SIZE, total);
        writeBatch(root, docs, batchStart, batchEnd, dimension);
        writer.writeBatch();
        root.clear();
      }
      writer.end();
    }
  }

  /**
   * Exports the given list of documents to an Arrow IPC file at {@code file}.
   *
   * @param docs the documents to export
   * @param dimension vector dimension
   * @param file destination file path (created or overwritten)
   * @throws IOException on I/O or serialization failure
   */
  public static void export(List<Document> docs, int dimension, Path file) throws IOException {
    try (OutputStream out = Files.newOutputStream(file)) {
      export(docs, dimension, out);
    }
  }

  // ---------------------------------------------------------------------------
  // Schema
  // ---------------------------------------------------------------------------

  static Schema buildSchema(int dimension) {
    Field id = new Field(COL_ID, FieldType.notNullable(new ArrowType.Utf8()), null);
    Field floatData =
        new Field(
            "$data$",
            FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
            null);
    Field vector =
        new Field(
            COL_VECTOR,
            FieldType.notNullable(new ArrowType.FixedSizeList(dimension)),
            List.of(floatData));
    Field text = new Field(COL_TEXT, FieldType.nullable(new ArrowType.Utf8()), null);
    Field metaJson = new Field(COL_METADATA_JSON, FieldType.nullable(new ArrowType.Utf8()), null);
    return new Schema(List.of(id, vector, text, metaJson));
  }

  // ---------------------------------------------------------------------------
  // Batch writer
  // ---------------------------------------------------------------------------

  private static void writeBatch(
      VectorSchemaRoot root, List<Document> docs, int batchStart, int batchEnd, int dimension) {

    VarCharVector idVec = (VarCharVector) root.getVector(COL_ID);
    FixedSizeListVector vectorVec = (FixedSizeListVector) root.getVector(COL_VECTOR);
    VarCharVector textVec = (VarCharVector) root.getVector(COL_TEXT);
    VarCharVector metaVec = (VarCharVector) root.getVector(COL_METADATA_JSON);
    Float4Vector dataVec = (Float4Vector) vectorVec.getDataVector();

    int batchSize = batchEnd - batchStart;
    idVec.allocateNew(batchSize);
    vectorVec.allocateNew();
    textVec.allocateNew(batchSize);
    metaVec.allocateNew(batchSize);
    dataVec.allocateNew(batchSize * dimension);

    for (int i = 0; i < batchSize; i++) {
      Document doc = docs.get(batchStart + i);
      int floatOffset = i * dimension;

      // id
      byte[] idBytes = doc.id().getBytes(UTF_8);
      idVec.setSafe(i, idBytes, 0, idBytes.length);

      // vector floats
      float[] v = doc.vector();
      vectorVec.setNotNull(i);
      for (int j = 0; j < dimension; j++) {
        dataVec.setSafe(floatOffset + j, v[j]);
      }

      // text (nullable)
      String txt = doc.text();
      if (txt != null) {
        byte[] tb = txt.getBytes(UTF_8);
        textVec.setSafe(i, tb, 0, tb.length);
      } else {
        textVec.setNull(i);
      }

      // metadata_json (nullable)
      Map<String, MetadataValue> meta = doc.metadata();
      if (meta != null && !meta.isEmpty()) {
        byte[] mb = toMetadataJson(meta).getBytes(UTF_8);
        metaVec.setSafe(i, mb, 0, mb.length);
      } else {
        metaVec.setNull(i);
      }
    }

    root.setRowCount(batchSize);
  }

  // ---------------------------------------------------------------------------
  // Lightweight metadata to JSON (no external library needed)
  // ---------------------------------------------------------------------------

  static String toMetadataJson(Map<String, MetadataValue> meta) {
    StringBuilder sb = new StringBuilder("{");
    boolean first = true;
    for (Map.Entry<String, MetadataValue> e : meta.entrySet()) {
      if (!first) sb.append(',');
      first = false;
      jsonString(sb, e.getKey());
      sb.append(':');
      appendValue(sb, e.getValue());
    }
    sb.append('}');
    return sb.toString();
  }

  private static void appendValue(StringBuilder sb, MetadataValue mv) {
    if (mv == null) {
      sb.append("null");
      return;
    }
    switch (mv) {
      case MetadataValue.Str s -> jsonString(sb, s.value());
      case MetadataValue.Num n -> sb.append(n.value());
      case MetadataValue.Bool b -> sb.append(b.value());
      case MetadataValue.Tags t -> {
        sb.append('[');
        List<String> items = t.values();
        for (int i = 0; i < items.size(); i++) {
          if (i > 0) sb.append(',');
          jsonString(sb, items.get(i));
        }
        sb.append(']');
      }
    }
  }

  private static void jsonString(StringBuilder sb, String s) {
    sb.append('"');
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '"' -> sb.append("\\\"");
        case '\\' -> sb.append("\\\\");
        case '\n' -> sb.append("\\n");
        case '\r' -> sb.append("\\r");
        case '\t' -> sb.append("\\t");
        case '\b' -> sb.append("\\b");
        case '\f' -> sb.append("\\f");
        default -> {
          if (c < 0x20) {
            // RFC 7159: control characters U+0000–U+001F must be escaped
            sb.append("\\u");
            sb.append(String.format("%04x", (int) c));
          } else {
            sb.append(c);
          }
        }
      }
    }
    sb.append('"');
  }
}
