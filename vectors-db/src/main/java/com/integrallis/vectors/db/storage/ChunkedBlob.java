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
package com.integrallis.vectors.db.storage;

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Objects;

/**
 * Ships and reads an arbitrarily large logical byte stream to/from object storage as a sequence of
 * fixed-size chunk objects — the shared machinery behind {@link ChunkedGraphBlob} (graph.bin) and
 * the chunked {@code quantized.bin} codes path. A logical stream produced by a {@link StreamWriter}
 * is split at {@link #CHUNK_SIZE} boundaries into objects named {@code <keyBase>.00000}, {@code
 * .00001}, … and read back lazily (one chunk resident at a time) so neither ship nor read ever
 * materializes a >2 GB {@code byte[]}.
 *
 * <p>Chunks are written contiguously from index 0, so the reader stops at the first {@link
 * StorageBackend#get(String)} that returns {@code null}. Generation directories are written fresh,
 * so there are never stale trailing chunks.
 */
public final class ChunkedBlob {

  /** Bytes per chunk object. 512 MiB is well under the 2 GB heap-array / single-PUT ceilings. */
  public static final int CHUNK_SIZE = 512 << 20;

  private ChunkedBlob() {}

  /** Writes to an {@link OutputStream} that the caller streams a logical blob into. */
  @FunctionalInterface
  public interface StreamWriter {
    void writeTo(OutputStream out) throws IOException;
  }

  static String chunkKey(String keyBase, int index) {
    return keyBase + "." + String.format("%05d", index);
  }

  /**
   * Streams {@code writer}'s output into chunk objects under {@code keyBase}. Returns the total
   * bytes written. Uses {@link #CHUNK_SIZE}.
   */
  public static long writeStream(StorageBackend backend, String keyBase, StreamWriter writer)
      throws IOException {
    return writeStream(backend, keyBase, writer, CHUNK_SIZE);
  }

  // Package-private: chunkSize is a parameter so tests can force multi-chunk output on tiny input.
  static long writeStream(
      StorageBackend backend, String keyBase, StreamWriter writer, int chunkSize)
      throws IOException {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(keyBase, "keyBase");
    Objects.requireNonNull(writer, "writer");
    try (ChunkingOutputStream out = new ChunkingOutputStream(backend, keyBase, chunkSize)) {
      writer.writeTo(out);
      out.finish();
      return out.totalBytes();
    }
  }

  /**
   * Opens a lazily-fetched {@link InputStream} over the chunk objects under {@code keyBase}, or
   * {@code null} if no {@code .00000} chunk exists (caller handles the legacy single-object case).
   */
  public static InputStream openStream(StorageBackend backend, String keyBase) throws IOException {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(keyBase, "keyBase");
    byte[] first = backend.get(chunkKey(keyBase, 0));
    return first == null ? null : new ChunkedInputStream(backend, keyBase, first);
  }

  /** Buffers up to {@code chunkSize} bytes, flushing each full buffer as the next chunk object. */
  private static final class ChunkingOutputStream extends OutputStream {
    private final StorageBackend backend;
    private final String keyBase;
    private final byte[] buf;
    private int pos;
    private int nextChunk;
    private long total;

    ChunkingOutputStream(StorageBackend backend, String keyBase, int chunkSize) {
      if (chunkSize <= 0) {
        throw new IllegalArgumentException("chunkSize must be positive: " + chunkSize);
      }
      this.backend = backend;
      this.keyBase = keyBase;
      this.buf = new byte[chunkSize];
    }

    long totalBytes() {
      return total;
    }

    @Override
    public void write(int b) throws IOException {
      if (pos == buf.length) {
        flushChunk();
      }
      buf[pos++] = (byte) b;
      total++;
    }

    @Override
    public void write(byte[] src, int off, int len) throws IOException {
      while (len > 0) {
        if (pos == buf.length) {
          flushChunk();
        }
        int n = Math.min(len, buf.length - pos);
        System.arraycopy(src, off, buf, pos, n);
        pos += n;
        off += n;
        len -= n;
        total += n;
      }
    }

    private void flushChunk() throws IOException {
      backend.put(chunkKey(keyBase, nextChunk++), Arrays.copyOf(buf, pos));
      pos = 0;
    }

    /** Writes the trailing partial chunk (and guarantees at least chunk 0 exists). */
    void finish() throws IOException {
      if (pos > 0 || nextChunk == 0) {
        flushChunk();
      }
    }

    @Override
    public void close() {
      // Chunks are flushed by finish(); nothing else to release.
    }
  }

  /** Lazily fetches chunk objects one at a time. */
  private static final class ChunkedInputStream extends InputStream {
    private final StorageBackend backend;
    private final String keyBase;
    private byte[] cur;
    private int pos;
    private int nextChunk;
    private boolean eof;

    ChunkedInputStream(StorageBackend backend, String keyBase, byte[] firstChunk) {
      this.backend = backend;
      this.keyBase = keyBase;
      this.cur = firstChunk;
      this.nextChunk = 1;
    }

    private boolean ensure() throws IOException {
      while (cur == null || pos >= cur.length) {
        if (eof) {
          return false;
        }
        cur = backend.get(chunkKey(keyBase, nextChunk++));
        pos = 0;
        if (cur == null) {
          eof = true;
          return false;
        }
      }
      return true;
    }

    @Override
    public int read() throws IOException {
      if (!ensure()) {
        return -1;
      }
      return cur[pos++] & 0xFF;
    }

    @Override
    public int read(byte[] dst, int off, int len) throws IOException {
      if (len == 0) {
        return 0;
      }
      if (!ensure()) {
        return -1;
      }
      int n = Math.min(len, cur.length - pos);
      System.arraycopy(cur, pos, dst, off, n);
      pos += n;
      return n;
    }
  }
}
