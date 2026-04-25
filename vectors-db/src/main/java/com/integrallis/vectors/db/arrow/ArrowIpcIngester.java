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

import com.integrallis.vectors.db.Document;
import com.integrallis.vectors.db.MetadataValue;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowStreamReader;

/**
 * Reads an Apache Arrow IPC stream (produced by {@link ArrowIpcExporter}) and reconstructs {@link
 * Document} objects.
 *
 * <p>Supported operations:
 *
 * <ul>
 *   <li>{@link #read(InputStream)} / {@link #read(Path)} — deserialize to a list of documents
 *   <li>{@link #ingest(VectorCollection, InputStream)} / {@link #ingest(VectorCollection, Path)} —
 *       deserialize and add all documents to a collection in a single batch
 * </ul>
 *
 * <p>The stream must conform to the schema produced by {@link ArrowIpcExporter#buildSchema(int)}
 * (columns: {@code id}, {@code vector}, {@code text}, {@code metadata_json}).
 */
public final class ArrowIpcIngester {

  private ArrowIpcIngester() {}

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Reads all documents from an Arrow IPC stream.
   *
   * @param in Arrow IPC stream; not closed by this method
   * @return mutable list of reconstructed documents
   * @throws IOException on read or deserialization failure
   */
  public static List<Document> read(InputStream in) throws IOException {
    List<Document> result = new ArrayList<>();
    try (BufferAllocator allocator = new RootAllocator();
        ArrowStreamReader reader = new ArrowStreamReader(Channels.newChannel(in), allocator)) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      while (reader.loadNextBatch()) {
        readBatch(root, result);
      }
    }
    return result;
  }

  /**
   * Reads all documents from an Arrow IPC file.
   *
   * @param file path to the Arrow IPC file
   * @return mutable list of reconstructed documents
   * @throws IOException on I/O or deserialization failure
   */
  public static List<Document> read(Path file) throws IOException {
    try (InputStream in = Files.newInputStream(file)) {
      return read(in);
    }
  }

  /**
   * Reads all documents from an Arrow IPC stream and adds them to {@code collection}.
   *
   * @param collection target collection (must already be configured with the matching dimension)
   * @param in Arrow IPC stream; not closed by this method
   * @return number of documents ingested
   * @throws IOException on read or deserialization failure
   */
  public static int ingest(VectorCollection collection, InputStream in) throws IOException {
    List<Document> docs = read(in);
    collection.addAll(docs);
    return docs.size();
  }

  /**
   * Reads all documents from an Arrow IPC file and adds them to {@code collection}.
   *
   * @param collection target collection
   * @param file path to the Arrow IPC file
   * @return number of documents ingested
   * @throws IOException on I/O or deserialization failure
   */
  public static int ingest(VectorCollection collection, Path file) throws IOException {
    try (InputStream in = Files.newInputStream(file)) {
      return ingest(collection, in);
    }
  }

  // ---------------------------------------------------------------------------
  // Batch reader
  // ---------------------------------------------------------------------------

  private static void readBatch(VectorSchemaRoot root, List<Document> out) {
    VarCharVector idVec = (VarCharVector) root.getVector(ArrowIpcExporter.COL_ID);
    FixedSizeListVector vectorVec =
        (FixedSizeListVector) root.getVector(ArrowIpcExporter.COL_VECTOR);
    VarCharVector textVec = (VarCharVector) root.getVector(ArrowIpcExporter.COL_TEXT);
    VarCharVector metaVec = (VarCharVector) root.getVector(ArrowIpcExporter.COL_METADATA_JSON);
    Float4Vector dataVec = (Float4Vector) vectorVec.getDataVector();

    int listSize = vectorVec.getListSize();
    int rows = root.getRowCount();

    for (int i = 0; i < rows; i++) {
      // id
      String id = new String(idVec.get(i), UTF_8);

      // vector
      int floatOffset = i * listSize;
      float[] v = new float[listSize];
      for (int j = 0; j < listSize; j++) {
        v[j] = dataVec.get(floatOffset + j);
      }

      // text (nullable)
      String text = textVec.isNull(i) ? null : new String(textVec.get(i), UTF_8);

      // metadata_json (nullable)
      Map<String, MetadataValue> meta;
      if (metaVec.isNull(i)) {
        meta = Map.of();
      } else {
        String json = new String(metaVec.get(i), UTF_8);
        meta = parseMetadataJson(json);
      }

      out.add(new Document(id, v, text, meta));
    }
  }

  // ---------------------------------------------------------------------------
  // Minimal JSON parser for metadata_json values produced by ArrowIpcExporter
  // ---------------------------------------------------------------------------

  /**
   * Parses a JSON object of the form produced by {@link ArrowIpcExporter#toMetadataJson} into a
   * {@link MetadataValue} map. Supports: string, number (stored as double), boolean, and
   * string-array values.
   */
  static Map<String, MetadataValue> parseMetadataJson(String json) {
    Map<String, MetadataValue> result = new LinkedHashMap<>();
    json = json.strip();
    if (json.isEmpty() || json.equals("{}")) return result;
    // Strip outer braces
    if (json.charAt(0) != '{' || json.charAt(json.length() - 1) != '}') return result;
    json = json.substring(1, json.length() - 1);

    int pos = 0;
    while (pos < json.length()) {
      pos = skipWhitespace(json, pos);
      if (pos >= json.length()) break;
      // Parse key
      if (json.charAt(pos) != '"') break;
      int[] keyEnd = new int[1];
      String key = parseString(json, pos, keyEnd);
      pos = keyEnd[0];
      pos = skipWhitespace(json, pos);
      if (pos >= json.length() || json.charAt(pos) != ':') break;
      pos++;
      pos = skipWhitespace(json, pos);
      // Parse value
      int[] valEnd = new int[1];
      MetadataValue mv = parseValue(json, pos, valEnd);
      pos = valEnd[0];
      if (mv != null) result.put(key, mv);
      pos = skipWhitespace(json, pos);
      if (pos < json.length() && json.charAt(pos) == ',') pos++;
    }
    return result;
  }

  private static MetadataValue parseValue(String json, int pos, int[] end) {
    if (pos >= json.length()) {
      end[0] = pos;
      return null;
    }
    char c = json.charAt(pos);
    if (c == '"') {
      int[] strEnd = new int[1];
      String s = parseString(json, pos, strEnd);
      end[0] = strEnd[0];
      return new MetadataValue.Str(s);
    } else if (c == '[') {
      // string array
      List<String> tags = new ArrayList<>();
      pos++;
      while (pos < json.length() && json.charAt(pos) != ']') {
        pos = skipWhitespace(json, pos);
        if (json.charAt(pos) == '"') {
          int[] strEnd = new int[1];
          tags.add(parseString(json, pos, strEnd));
          pos = strEnd[0];
        }
        pos = skipWhitespace(json, pos);
        if (pos < json.length() && json.charAt(pos) == ',') pos++;
      }
      end[0] = pos + 1;
      return new MetadataValue.Tags(tags);
    } else if (json.startsWith("true", pos)) {
      end[0] = pos + 4;
      return new MetadataValue.Bool(true);
    } else if (json.startsWith("false", pos)) {
      end[0] = pos + 5;
      return new MetadataValue.Bool(false);
    } else if (json.startsWith("null", pos)) {
      end[0] = pos + 4;
      return null;
    } else {
      // number
      int start = pos;
      while (pos < json.length() && ",}]".indexOf(json.charAt(pos)) < 0) pos++;
      double d = Double.parseDouble(json.substring(start, pos).strip());
      end[0] = pos;
      return new MetadataValue.Num(d);
    }
  }

  private static String parseString(String json, int pos, int[] end) {
    pos++; // skip opening quote
    StringBuilder sb = new StringBuilder();
    while (pos < json.length()) {
      char c = json.charAt(pos++);
      if (c == '"') break;
      if (c == '\\' && pos < json.length()) {
        char esc = json.charAt(pos++);
        switch (esc) {
          case 'n' -> sb.append('\n');
          case 'r' -> sb.append('\r');
          case 't' -> sb.append('\t');
          case 'b' -> sb.append('\b');
          case 'f' -> sb.append('\f');
          case 'u' -> {
            // JSON unicode escape (backslash-u followed by 4 hex digits)
            if (pos + 4 <= json.length()) {
              int codePoint = Integer.parseInt(json.substring(pos, pos + 4), 16);
              sb.append((char) codePoint);
              pos += 4;
            }
          }
          default -> sb.append(esc); // handles \" \\ and unknown sequences
        }
      } else {
        sb.append(c);
      }
    }
    end[0] = pos;
    return sb.toString();
  }

  private static int skipWhitespace(String json, int pos) {
    while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) pos++;
    return pos;
  }
}
