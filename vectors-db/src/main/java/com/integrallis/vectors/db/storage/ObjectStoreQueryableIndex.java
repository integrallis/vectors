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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.index.IndexSpi.SearchOutcome;
import com.integrallis.vectors.hnsw.HnswGraph;
import com.integrallis.vectors.quantization.CompressedVectors;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Queries a committed generation <b>directly from object storage</b>, with no local hydration — the
 * SOTA object-storage index query entry point. It pulls the small blobs (graph adjacency +
 * Ext-RaBitQ codes) into RAM once and serves the full float32 vectors on demand via ranged GETs
 * ({@link ObjectStoreRandomAccessVectors}); navigation runs entirely on the RAM codes and only the
 * rerank fetches (over-query × k, parallelized) touch object storage.
 *
 * <p>Reads the layout that {@code GenerationShippingSubscriber} writes: {@code <prefix>CURRENT} (an
 * 8-byte little-endian generation number) points at {@code <prefix>gen-NNNN/}, which holds {@code
 * graph.bin}, {@code quantized.bin} and {@code vectors.bin}. Count and dimension are derived from
 * the decoded codes, so the caller supplies only the backend, prefix, and metric.
 *
 * <p>Requires a quantized HNSW generation — the object-storage query path navigates on codes. A
 * generation without {@code quantized.bin} cannot be queried this way (fetching every vector for
 * navigation would defeat the design).
 */
public final class ObjectStoreQueryableIndex implements AutoCloseable {

  private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

  private final MappedHnswIndexAdapter adapter;
  private final int dimension;
  private final int size;

  private ObjectStoreQueryableIndex(MappedHnswIndexAdapter adapter, int dimension, int size) {
    this.adapter = adapter;
    this.dimension = dimension;
    this.size = size;
  }

  /**
   * Opens the generation referenced by {@code <prefix>CURRENT} for object-storage querying.
   *
   * @param backend the object-storage backend holding the shipped generation
   * @param prefix the key prefix the generation was shipped under ({@code ""} for the bucket root)
   * @param metric the similarity metric the graph was built with
   * @return a queryable index reading directly from object storage
   * @throws IOException if {@code CURRENT} or the generation blobs are missing/unreadable
   */
  public static ObjectStoreQueryableIndex openCurrent(
      StorageBackend backend, String prefix, SimilarityFunction metric) throws IOException {
    String p = normalizePrefix(prefix);
    byte[] current = backend.get(p + FileFormat.CURRENT_FILE);
    if (current == null || current.length < Long.BYTES) {
      throw new IOException("no valid " + FileFormat.CURRENT_FILE + " under prefix '" + p + "'");
    }
    long gen = ByteBuffer.wrap(current).order(LE).getLong();
    return open(backend, p + FileFormat.generationDirName(gen) + "/", metric);
  }

  /**
   * Opens a specific generation (by its full key prefix, e.g. {@code some/prefix/gen-000…0000/}).
   *
   * @param backend the object-storage backend
   * @param generationPrefix the key prefix of the generation directory (must end in {@code /})
   * @param metric the similarity metric the graph was built with
   * @return a queryable index reading directly from object storage
   * @throws IOException if the generation's graph or quantized blob is missing
   */
  public static ObjectStoreQueryableIndex open(
      StorageBackend backend, String generationPrefix, SimilarityFunction metric)
      throws IOException {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(metric, "metric");
    String gp = generationPrefix.endsWith("/") ? generationPrefix : generationPrefix + "/";

    byte[] graphBytes = backend.get(gp + FileFormat.GRAPH_FILE);
    if (graphBytes == null) {
      throw new IOException("missing " + gp + FileFormat.GRAPH_FILE);
    }
    HnswGraph graph = HnswGraphCodec.decode(graphBytes);

    byte[] quantizedBytes = backend.get(gp + FileFormat.QUANTIZED_FILE);
    if (quantizedBytes == null) {
      throw new IOException(
          "object-storage query requires quantization; missing " + gp + FileFormat.QUANTIZED_FILE);
    }
    CompressedVectors codes = QuantizedVectorsCodec.decode(quantizedBytes);

    int size = codes.size();
    int dimension = codes.dimension();
    ObjectStoreRandomAccessVectors vectors =
        new ObjectStoreRandomAccessVectors(backend, gp + FileFormat.VECTORS_FILE, size, dimension);

    MappedHnswIndexAdapter adapter = new MappedHnswIndexAdapter(graph, vectors, metric);
    adapter.enableQuantization(codes);
    return new ObjectStoreQueryableIndex(adapter, dimension, size);
  }

  /**
   * Two-pass search: navigate on the RAM-resident codes, then rerank the over-query candidate set
   * in full precision via parallel ranged GETs. Use {@code overQueryFactor > 1} to engage the
   * two-pass path (otherwise navigation-only quantized results are returned).
   */
  public SearchOutcome search(float[] query, int k, int efSearch, float overQueryFactor) {
    return adapter.search(query, k, efSearch, overQueryFactor);
  }

  /** Number of vectors in the generation. */
  public int size() {
    return size;
  }

  /** Vector dimensionality. */
  public int dimension() {
    return dimension;
  }

  @Override
  public void close() {
    adapter.close();
  }

  private static String normalizePrefix(String prefix) {
    if (prefix == null || prefix.isEmpty()) {
      return "";
    }
    return prefix.endsWith("/") ? prefix : prefix + "/";
  }
}
