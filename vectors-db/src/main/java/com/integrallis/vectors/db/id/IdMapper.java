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
package com.integrallis.vectors.db.id;

/**
 * Bidirectional mapping between external string ids and dense integer ordinals.
 *
 * <p>Ordinals are assigned sequentially starting from 0. {@link #put(String)} is strict: duplicates
 * throw because the v0.1 {@code add} contract is insert-only. {@code upsert} lands in Step 6.
 *
 * <p><b>Thread safety.</b> Implementations are <i>not</i> required to be thread-safe on their own.
 * The facade ({@code VectorCollectionImpl}) protects a mapper instance by mutating it only under
 * the writer lock and publishing a fully-populated successor via a volatile {@code Generation}
 * record — readers capture one snapshot and never observe torn state.
 *
 * <p>Step 4a introduces a mapped implementation where {@link #put(String)} throws — persistent
 * generations are written whole-file via a static {@code Writer.writeTo(...)} helper, not by
 * per-entry insertion.
 */
public interface IdMapper extends AutoCloseable {

  /**
   * Registers an external id and returns its assigned ordinal.
   *
   * @throws IllegalArgumentException if the id has already been registered
   * @throws UnsupportedOperationException if this mapper is read-only (mapped implementations)
   */
  int put(String id);

  /** Returns {@code true} if the external id is known to this mapper. */
  boolean contains(String id);

  /** Returns the ordinal assigned to {@code id}, or {@code -1} if unknown. */
  int ordinalOf(String id);

  /**
   * Returns the external id for the given ordinal.
   *
   * @throws IndexOutOfBoundsException if {@code ordinal} is out of range
   */
  String idOf(int ordinal);

  /** Number of mappings currently stored. */
  int size();

  /**
   * Closes underlying resources (e.g. mmap arena for the mapped implementation). In-memory
   * implementations default to a no-op; the enclosing {@code Generation} is responsible for
   * lifecycle sequencing.
   */
  @Override
  default void close() {
    // no-op for in-memory; mapped variants override if they own resources
  }
}
