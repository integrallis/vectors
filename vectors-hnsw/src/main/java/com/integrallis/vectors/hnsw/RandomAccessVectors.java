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
package com.integrallis.vectors.hnsw;

/**
 * Read-only access to vectors by ordinal. Decoupled from the quantization module's {@code
 * VectorDataset} to keep the HNSW module dependency-light.
 *
 * <p>Implementations may return a shared buffer from {@link #getVector(int)}; callers must not
 * retain the returned array across calls.
 */
public interface RandomAccessVectors {

  /** Returns the number of vectors. */
  int size();

  /** Returns the number of dimensions per vector. */
  int dimension();

  /**
   * Returns the vector at the given ordinal. The returned array may be a shared buffer; callers
   * must not retain it across calls.
   *
   * @param ordinal the 0-based vector index
   * @return the vector data
   */
  float[] getVector(int ordinal);

  /**
   * {@code true} when every call to {@link #getVector(int)} may overwrite the buffer returned by
   * previous calls (e.g. mmap-backed implementations with a single scratch row). Callers that want
   * to gather multiple vectors into a reusable {@code float[][]} pool before a fused SIMD scan must
   * copy the returned data (or skip the bulk path) when this is {@code true}.
   *
   * <p>The default is {@code true} (safe assumption). Implementations backed by a stable {@code
   * float[][]} (e.g. {@code InMemoryVectors}) should override this to return {@code false}.
   */
  default boolean sharesReturnBuffer() {
    return true;
  }

  /**
   * {@code true} when this implementation can hand out a zero-copy {@link
   * java.lang.foreign.MemorySegment} view of a stored vector via {@link #vectorSegment(int)}.
   * Off-heap / mmap-backed implementations that store contiguous little-endian float32 rows should
   * override this to return {@code true}, enabling the searcher to SIMD-score directly from the
   * segment with no intermediate {@code float[]} copy.
   *
   * <p>The default is {@code false} — heap-backed implementations (e.g. {@code InMemoryVectors})
   * have no off-heap view to offer and stay on the {@code float[]} path.
   */
  default boolean supportsSegments() {
    return false;
  }

  /**
   * Returns a zero-copy {@link java.lang.foreign.MemorySegment} view of the vector at the given
   * ordinal, or {@code null} when {@link #supportsSegments()} is {@code false}. The segment covers
   * exactly {@code dimension() * 4} bytes of little-endian float32 data and is intended to be fed
   * directly to {@code VectorUtil.dotProduct(MemorySegment, MemorySegment, int)} (and friends).
   *
   * <p><b>Lifetime.</b> When an implementation reuses a single scratch view, the returned segment
   * is only valid until the next {@link #vectorSegment(int)} access on the same thread. However,
   * slices into an immutable mmap'd region (the intended off-heap backing) are <i>stable</i>:
   * distinct ordinals return distinct, independently valid views that may be held and compared
   * freely for as long as the backing arena is open. Callers on the search path score-then-discard
   * immediately, so either contract is satisfied.
   *
   * @param ordinal the 0-based vector index
   * @return a zero-copy segment view, or {@code null} if segments are unsupported
   */
  default java.lang.foreign.MemorySegment vectorSegment(int ordinal) {
    return null;
  }
}
