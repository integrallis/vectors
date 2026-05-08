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
package com.integrallis.vectors.ingest.cursor;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.vectors.ingest.IngestCursor;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ConcurrentModificationException;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Durable {@link IngestCursor} that persists per-source offsets as a single small JSON object on a
 * {@link StorageBackend}. Reads and writes happen on the configured {@code key} (default {@code
 * _ingest/cursor.json}); each save is a single {@link StorageBackend#conditionalPut conditionalPut}
 * that supplies the etag returned by the previous successful save, so concurrent writers
 * race-detect via CAS.
 *
 * <p>Limitations:
 *
 * <ul>
 *   <li><b>Single-writer per source.</b> When two cursor instances start fresh against an
 *       already-existing key, the very first save in each process cannot CAS-validate (the backend
 *       exposes no etag-on-get); the cursor falls back to an unconditional {@link
 *       StorageBackend#put put}. Subsequent saves use full CAS and throw {@link
 *       ConcurrentModificationException} on conflict.
 * </ul>
 */
public final class R2KeyCursor implements IngestCursor {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** Default key. */
  public static final String DEFAULT_KEY = "_ingest/cursor.json";

  private final StorageBackend backend;
  private final String key;
  private final ConcurrentHashMap<String, Long> offsets = new ConcurrentHashMap<>();
  private final Object writeLock = new Object();
  private volatile String etag;
  private volatile boolean loaded;

  public R2KeyCursor(StorageBackend backend) {
    this(backend, DEFAULT_KEY);
  }

  public R2KeyCursor(StorageBackend backend, String key) {
    this.backend = Objects.requireNonNull(backend, "backend");
    if (key == null || key.isBlank()) {
      throw new IllegalArgumentException("key must be non-blank");
    }
    this.key = key;
  }

  @Override
  public long load(String sourceName) throws IOException {
    Objects.requireNonNull(sourceName, "sourceName");
    ensureLoaded();
    return offsets.getOrDefault(sourceName, 0L);
  }

  @Override
  public void save(String sourceName, long offset) throws IOException {
    Objects.requireNonNull(sourceName, "sourceName");
    if (offset < 0) {
      throw new IllegalArgumentException("offset must be >= 0");
    }
    synchronized (writeLock) {
      ensureLoaded();
      offsets.put(sourceName, offset);
      byte[] payload = serialise();
      if (etag != null) {
        StorageBackend.ConditionalPutResult r = backend.conditionalPut(key, payload, etag);
        if (!r.succeeded()) {
          throw new ConcurrentModificationException(
              "cursor key '" + key + "' was modified by another writer");
        }
        etag = r.newEtag();
      } else {
        // First save in this process. Try CAS-null (key must be absent); fall back to unconditional
        // put if the key already existed when we loaded.
        StorageBackend.ConditionalPutResult r = backend.conditionalPut(key, payload, null);
        if (r.succeeded()) {
          etag = r.newEtag();
        } else {
          backend.put(key, payload);
          // We cannot recover the etag without a separate read API; subsequent saves will retry the
          // CAS-null path until they succeed (or fall back again).
        }
      }
    }
  }

  private void ensureLoaded() throws IOException {
    if (loaded) return;
    synchronized (writeLock) {
      if (loaded) return;
      byte[] data = backend.get(key);
      if (data != null) {
        JsonNode root = MAPPER.readTree(data);
        if (root.isObject()) {
          for (var e : root.properties()) {
            if (e.getValue().isIntegralNumber()) {
              offsets.put(e.getKey(), e.getValue().asLong());
            }
          }
        }
      }
      loaded = true;
    }
  }

  private byte[] serialise() {
    ObjectNode root = MAPPER.createObjectNode();
    offsets.forEach(root::put);
    try {
      return MAPPER.writeValueAsBytes(root);
    } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
      throw new IllegalStateException("cursor JSON serialisation failed", e);
    }
  }

  // ─── visible for tests ─────────────────────────────────────────────────────

  String currentEtag() {
    return etag;
  }

  /** Returns the storage key the cursor reads/writes. */
  public String key() {
    return key;
  }

  /** Encoded UTF-8 string view of the cursor JSON for diagnostics. */
  public String snapshotJson() {
    return new String(serialise(), StandardCharsets.UTF_8);
  }
}
