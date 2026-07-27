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
package com.integrallis.vectors.ingest.sources;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.ingest.IngestDoc;
import com.integrallis.vectors.ingest.IngestSource;
import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.LongConsumer;

/**
 * Reads documents from a JSON-lines file. Each non-blank line must be a JSON object with at least
 * an {@code id} and one of {@code text} / {@code precomputedVector}; {@code mime} and a string
 * {@code attrs} object are optional.
 *
 * <p>By default, malformed lines abort iteration. Pass {@code skipMalformed=true} to log-and-skip;
 * an optional {@code malformedLineObserver} receives the offending line numbers (1-based).
 */
public final class JsonlSource implements IngestSource {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final String name;
  private final Path file;
  private final long startOffset;
  private final boolean skipMalformed;
  private final LongConsumer malformedLineObserver;

  public JsonlSource(String name, Path file) {
    this(name, file, 0L, false, n -> {});
  }

  public JsonlSource(
      String name,
      Path file,
      long startOffset,
      boolean skipMalformed,
      LongConsumer malformedLineObserver) {
    Objects.requireNonNull(name, "name");
    if (name.isBlank()) {
      throw new IllegalArgumentException("name must be non-blank");
    }
    Objects.requireNonNull(file, "file");
    if (startOffset < 0) {
      throw new IllegalArgumentException("startOffset must be >= 0");
    }
    this.name = name;
    this.file = file;
    this.startOffset = startOffset;
    this.skipMalformed = skipMalformed;
    this.malformedLineObserver = malformedLineObserver != null ? malformedLineObserver : n -> {};
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public OptionalLong estimatedSize() {
    return OptionalLong.empty();
  }

  @Override
  public long startOffset() {
    return startOffset;
  }

  @Override
  public Iterator<IngestDoc> iterator() {
    BufferedReader reader;
    try {
      reader = Files.newBufferedReader(file, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return new Iter(reader, startOffset, skipMalformed, malformedLineObserver);
  }

  /**
   * Iterator over the JSONL file. Implements {@link Closeable} so a caller that abandons iteration
   * early (a chunk/batch limit, a downstream error, a {@code break}) can release the file
   * descriptor deterministically — the previous behavior closed the reader only on natural EOF or
   * an {@code IOException}, leaking one FD per aborted source. The pipeline closes it in a {@code
   * finally}; {@link #close()} is idempotent and, once closed, {@link #hasNext()} short-circuits to
   * {@code false} rather than touching the closed reader.
   */
  private static final class Iter implements Iterator<IngestDoc>, Closeable {
    private final BufferedReader reader;
    private final boolean skipMalformed;
    private final LongConsumer malformedObserver;
    private long lineNumber;
    private long emitted;
    private long skipRemaining;
    private IngestDoc next;
    private boolean closed;

    Iter(BufferedReader reader, long skip, boolean skipMalformed, LongConsumer obs) {
      this.reader = reader;
      this.skipRemaining = skip;
      this.skipMalformed = skipMalformed;
      this.malformedObserver = obs;
    }

    @Override
    public boolean hasNext() {
      if (next != null) return true;
      if (closed) return false;
      try {
        String line;
        while ((line = reader.readLine()) != null) {
          lineNumber++;
          if (line.isBlank()) continue;
          IngestDoc parsed;
          try {
            parsed = parseLine(line);
          } catch (Exception e) {
            if (skipMalformed) {
              malformedObserver.accept(lineNumber);
              continue;
            }
            throw new IOException("malformed JSONL at line " + lineNumber + ": " + line, e);
          }
          if (skipRemaining > 0) {
            skipRemaining--;
            continue;
          }
          emitted++;
          next = parsed;
          return true;
        }
        closeQuietly();
        return false;
      } catch (IOException e) {
        closeQuietly();
        throw new UncheckedIOException(e);
      }
    }

    @Override
    public IngestDoc next() {
      if (!hasNext()) throw new NoSuchElementException();
      IngestDoc out = next;
      next = null;
      return out;
    }

    @Override
    public void close() throws IOException {
      if (closed) return;
      closed = true;
      reader.close();
    }

    /** Idempotent best-effort close used on the internal EOF / error paths. */
    private void closeQuietly() {
      if (closed) return;
      closed = true;
      try {
        reader.close();
      } catch (IOException ignored) {
        // best effort — the caller is already handling EOF or propagating the original IOException
      }
    }
  }

  private static IngestDoc parseLine(String line) throws IOException {
    JsonNode node = MAPPER.readTree(line);
    if (!node.isObject()) {
      throw new IOException("expected JSON object");
    }
    String id = textOrNull(node.get("id"));
    if (id == null || id.isBlank()) {
      throw new IOException("missing 'id'");
    }
    String text = textOrNull(node.get("text"));
    String mime = textOrNull(node.get("mime"));
    Map<String, String> attrs = null;
    JsonNode attrsNode = node.get("attrs");
    if (attrsNode != null && attrsNode.isObject()) {
      attrs = new LinkedHashMap<>();
      for (Map.Entry<String, JsonNode> e : attrsNode.properties()) {
        attrs.put(e.getKey(), e.getValue().isNull() ? null : e.getValue().asText());
      }
    }
    float[] vec = null;
    JsonNode vecNode = node.get("precomputedVector");
    if (vecNode != null && vecNode.isArray()) {
      vec = new float[vecNode.size()];
      for (int i = 0; i < vec.length; i++) vec[i] = (float) vecNode.get(i).asDouble();
    }
    return new IngestDoc(id, text, null, mime != null ? mime : "text/plain", attrs, vec);
  }

  private static String textOrNull(JsonNode n) {
    return (n == null || n.isNull()) ? null : n.asText();
  }
}
