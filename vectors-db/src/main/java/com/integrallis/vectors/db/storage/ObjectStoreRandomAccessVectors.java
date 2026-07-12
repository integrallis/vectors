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
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Object-storage-backed {@link com.integrallis.vectors.hnsw.RandomAccessVectors} — the query-time
 * full-vector tier of the SOTA object-storage index. Full float32 vectors stay on object storage
 * ({@code vectors.bin}); each {@link #getVector(int)} pulls exactly one vector via a {@linkplain
 * StorageBackend#getRange ranged GET}. This is the drop-in remote counterpart to {@link
 * MemorySegmentRandomAccessVectors} (local mmap): the graph adjacency and the compact Ext-RaBitQ
 * codes stay resident in RAM for navigation, and the only object-storage reads are the rerank
 * fetches for the small over-query candidate set — the design validated in P1–P7.
 *
 * <p><b>Layout parity.</b> {@code vectors.bin} is headerless, row-major, little-endian float32 with
 * each vector padded to a 64-byte-aligned slot: {@code stride = alignUp(dimension * 4, 64)}, so the
 * vector at {@code ordinal} occupies {@code [ordinal * stride, ordinal * stride + dimension * 4)}
 * (the trailing bytes to {@code stride} are alignment padding and are not fetched). This matches
 * {@link MemorySegmentVectors} verbatim, so a generation reads identically whether mmapped or
 * served over the network.
 *
 * <p><b>Shared-buffer invariant.</b> {@link #getVector(int)} returns a per-thread scratch {@code
 * float[dimension]} overwritten on every call — same contract as {@link
 * MemorySegmentRandomAccessVectors}. Callers must not retain the array across a subsequent call on
 * the same thread. Because reads hit the network, callers on the hot path should batch the rerank
 * candidate fetches rather than issue one blocking {@code getRange} per candidate serially.
 *
 * <p>Implements both the {@code hnsw} and {@code vamana} {@code RandomAccessVectors} interfaces
 * (they are structurally identical) so it is a drop-in for either paged index adapter.
 */
public final class ObjectStoreRandomAccessVectors
    implements com.integrallis.vectors.hnsw.RandomAccessVectors,
        com.integrallis.vectors.vamana.RandomAccessVectors {

  private final StorageBackend backend;
  private final String key;
  private final int size;
  private final int dimension;
  private final long stride;
  private final int rawVectorByteSize;
  // Overwritten on every getVector() call on this thread; callers MUST NOT retain across calls.
  private final ThreadLocal<float[]> scratch;

  /**
   * @param backend the object-storage backend serving {@code vectors.bin}
   * @param key the object key for the vectors blob (e.g. {@code "vectors.bin"} within the
   *     generation)
   * @param size number of vectors stored
   * @param dimension float32 components per vector
   * @throws IllegalArgumentException if {@code size < 0} or {@code dimension < 1}
   */
  public ObjectStoreRandomAccessVectors(
      StorageBackend backend, String key, int size, int dimension) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.key = Objects.requireNonNull(key, "key");
    if (size < 0) {
      throw new IllegalArgumentException("size must be >= 0: " + size);
    }
    if (dimension < 1) {
      throw new IllegalArgumentException("dimension must be >= 1: " + dimension);
    }
    this.size = size;
    this.dimension = dimension;
    this.stride =
        AlignmentUtil.alignUp((long) dimension * Float.BYTES, AlignmentUtil.VECTOR_ALIGNMENT);
    this.rawVectorByteSize = dimension * Float.BYTES;
    this.scratch = ThreadLocal.withInitial(() -> new float[dimension]);
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  /**
   * Fetches the vector at {@code ordinal} via a single ranged GET and decodes it (little-endian
   * float32) into this thread's scratch buffer.
   *
   * @param ordinal the 0-based vector index
   * @return a per-thread {@code float[dimension]} — do NOT retain across subsequent calls on this
   *     thread
   * @throws IndexOutOfBoundsException if {@code ordinal} is out of range
   * @throws UncheckedIOException if the backend read fails or the object is missing
   */
  @Override
  public float[] getVector(int ordinal) {
    Objects.checkIndex(ordinal, size);
    byte[] raw;
    try {
      raw = backend.getRange(key, ordinal * stride, rawVectorByteSize);
    } catch (IOException e) {
      throw new UncheckedIOException("ranged GET failed for " + key + " ordinal " + ordinal, e);
    }
    if (raw == null) {
      throw new UncheckedIOException(
          "object not found: " + key, new IOException("getRange returned null"));
    }
    float[] dst = scratch.get();
    ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(dst);
    return dst;
  }

  /**
   * Per-thread scratch buffer is overwritten on every call — resolves the dual-interface default.
   */
  @Override
  public boolean sharesReturnBuffer() {
    return true;
  }

  /** No stable zero-copy segment over a ranged GET; rerank uses the {@code float[]} path. */
  @Override
  public boolean supportsSegments() {
    return false;
  }

  /** Segments are unsupported for network-backed reads; returns {@code null} per the contract. */
  @Override
  public MemorySegment vectorSegment(int ordinal) {
    return null;
  }
}
