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
package com.integrallis.vectors.storage.manifest;

import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * The authoritative pointer for an object-storage collection or cluster: a small, versioned object
 * advanced only by a compare-and-swap ({@link ManifestStore}). It carries a strictly-monotonic
 * {@code generation} (a total order that lets a stale/reordered publish be rejected — the object
 * storage analogue of the local-filesystem {@code CURRENT} pointer), an opaque {@code contentHash}
 * identifying the committed state, and a map of named {@code entries} to their committed
 * generation.
 *
 * <p>A single-shard collection has one entry; a sharded cluster lists one entry per shard, so a
 * single CAS of this manifest atomically publishes every shard's new generation together (the
 * commit point). See {@code java-ai/vectors-phase4-manifest-cas-design-2026-07-13.md}.
 *
 * <p>Immutable; {@link #encode()} / {@link #decode(byte[])} use a self-describing binary format
 * (magic + version) so it round-trips without any external serialization dependency.
 */
public record StorageManifest(
    long generation,
    String contentHash,
    Map<String, Long> entries,
    long committedAtEpochMs,
    String writer) {

  private static final int MAGIC = 0x564D414E; // "VMAN"
  private static final byte FORMAT_VERSION = 1;

  /** The sentinel for "no manifest yet": generation {@link Long#MIN_VALUE}, no entries. */
  public static final StorageManifest EMPTY =
      new StorageManifest(Long.MIN_VALUE, "", Map.of(), 0L, "");

  public StorageManifest {
    Objects.requireNonNull(contentHash, "contentHash");
    Objects.requireNonNull(writer, "writer");
    Objects.requireNonNull(entries, "entries");
    // Immutable, key-sorted copy: a stable iteration order makes encode() (and therefore the object
    // etag / content hash) deterministic across runs, which the CAS pointer semantics rely on.
    entries = Collections.unmodifiableMap(new TreeMap<>(entries));
  }

  /**
   * Returns a copy at the given generation (used by {@link ManifestStore#commit} to stamp gen+1).
   */
  public StorageManifest atGeneration(long newGeneration) {
    return new StorageManifest(newGeneration, contentHash, entries, committedAtEpochMs, writer);
  }

  /** Returns a copy with {@code key} mapped to {@code value} (added or replaced). */
  public StorageManifest withEntry(String key, long value) {
    Objects.requireNonNull(key, "key");
    Map<String, Long> next = new LinkedHashMap<>(entries);
    next.put(key, value);
    return new StorageManifest(generation, contentHash, next, committedAtEpochMs, writer);
  }

  /** Returns a copy carrying the given content hash, commit timestamp, and writer. */
  public StorageManifest withProvenance(String hash, long committedAtEpochMs, String writer) {
    return new StorageManifest(generation, hash, entries, committedAtEpochMs, writer);
  }

  /** {@code true} if no real manifest has been committed yet (generation is the sentinel). */
  public boolean isEmpty() {
    return generation == Long.MIN_VALUE;
  }

  /** Serialises to the versioned binary format. */
  public byte[] encode() {
    byte[] hashBytes = contentHash.getBytes(StandardCharsets.UTF_8);
    byte[] writerBytes = writer.getBytes(StandardCharsets.UTF_8);
    int size = 4 + 1 + 8 + 8 + 4 + hashBytes.length + 4 + writerBytes.length + 4;
    // Precompute entry key bytes so we size the buffer exactly.
    Map<byte[], Long> encodedEntries = new LinkedHashMap<>();
    for (Map.Entry<String, Long> e : entries.entrySet()) {
      byte[] k = e.getKey().getBytes(StandardCharsets.UTF_8);
      encodedEntries.put(k, e.getValue());
      size += 4 + k.length + 8;
    }
    ByteBuffer buf = ByteBuffer.allocate(size);
    buf.putInt(MAGIC);
    buf.put(FORMAT_VERSION);
    buf.putLong(generation);
    buf.putLong(committedAtEpochMs);
    buf.putInt(hashBytes.length).put(hashBytes);
    buf.putInt(writerBytes.length).put(writerBytes);
    buf.putInt(encodedEntries.size());
    for (Map.Entry<byte[], Long> e : encodedEntries.entrySet()) {
      buf.putInt(e.getKey().length).put(e.getKey()).putLong(e.getValue());
    }
    return buf.array();
  }

  /** Parses a manifest from {@link #encode()} bytes. */
  public static StorageManifest decode(byte[] bytes) {
    Objects.requireNonNull(bytes, "bytes");
    ByteBuffer buf = ByteBuffer.wrap(bytes);
    try {
      int magic = buf.getInt();
      if (magic != MAGIC) {
        throw new IllegalArgumentException(
            "not a StorageManifest (bad magic 0x" + Integer.toHexString(magic) + ")");
      }
      byte version = buf.get();
      if (version != FORMAT_VERSION) {
        throw new IllegalArgumentException("unsupported manifest format version: " + version);
      }
      long generation = buf.getLong();
      long committedAt = buf.getLong();
      String hash = readString(buf);
      String writer = readString(buf);
      int count = buf.getInt();
      if (count < 0) {
        throw new IllegalArgumentException("negative entry count: " + count);
      }
      Map<String, Long> entries = new LinkedHashMap<>();
      for (int i = 0; i < count; i++) {
        String key = readString(buf);
        entries.put(key, buf.getLong());
      }
      return new StorageManifest(generation, hash, entries, committedAt, writer);
    } catch (BufferUnderflowException e) {
      throw new IllegalArgumentException("truncated StorageManifest", e);
    }
  }

  private static String readString(ByteBuffer buf) {
    int len = buf.getInt();
    if (len < 0 || len > buf.remaining()) {
      throw new IllegalArgumentException("invalid string length in manifest: " + len);
    }
    byte[] b = new byte[len];
    buf.get(b);
    return new String(b, StandardCharsets.UTF_8);
  }
}
