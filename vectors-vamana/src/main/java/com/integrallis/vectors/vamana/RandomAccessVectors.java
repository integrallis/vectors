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
package com.integrallis.vectors.vamana;

/**
 * Read-only access to vectors by ordinal. Decoupled from the quantization module's {@code
 * VectorDataset} to keep the Vamana module dependency-light.
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
}
