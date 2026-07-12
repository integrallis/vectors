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

import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/**
 * Ships and reads an {@link HnswGraph} to/from object storage as chunk objects, sidestepping the ~2
 * GB ceiling of a single {@link StorageBackend#put(String, byte[])}. Thin adapter over {@link
 * ChunkedBlob}: the graph is streamed through {@link HnswGraphCodec#encode(HnswGraph,
 * java.io.OutputStream)} into {@code <prefix>graph.bin.00000}, {@code .00001}, … and decoded back
 * from a lazily-fetched concatenation of those chunks. For backward compatibility the reader falls
 * back to a single legacy {@code graph.bin} object when no {@code .00000} chunk exists.
 */
public final class ChunkedGraphBlob {

  /** Bytes per chunk object (see {@link ChunkedBlob#CHUNK_SIZE}). */
  public static final int CHUNK_SIZE = ChunkedBlob.CHUNK_SIZE;

  private ChunkedGraphBlob() {}

  /**
   * Streams {@code graph} into chunk objects under {@code keyPrefix + "graph.bin"}. Returns the
   * total encoded byte count (what {@link HnswGraphCodec#encode(HnswGraph)} would report).
   */
  public static long writeGraph(StorageBackend backend, String keyPrefix, HnswGraph graph)
      throws IOException {
    return writeGraph(backend, keyPrefix, graph, CHUNK_SIZE);
  }

  static long writeGraph(StorageBackend backend, String keyPrefix, HnswGraph graph, int chunkSize)
      throws IOException {
    Objects.requireNonNull(graph, "graph");
    String base = keyPrefix + FileFormat.GRAPH_FILE;
    return ChunkedBlob.writeStream(backend, base, out -> HnswGraphCodec.encode(graph, out), chunkSize);
  }

  /**
   * Decodes the graph under {@code keyPrefix + "graph.bin"}, reading chunk objects lazily. Returns
   * {@code null} if no graph is present (neither a {@code .00000} chunk nor a legacy single object).
   */
  public static HnswGraph openGraph(StorageBackend backend, String keyPrefix) throws IOException {
    Objects.requireNonNull(backend, "backend");
    String base = keyPrefix + FileFormat.GRAPH_FILE;
    InputStream chunks = ChunkedBlob.openStream(backend, base);
    if (chunks == null) {
      byte[] single = backend.get(base); // legacy single-object graph.bin
      return single == null ? null : HnswGraphCodec.decode(single);
    }
    try (InputStream in = chunks) {
      return HnswGraphCodec.decode(in);
    }
  }
}
