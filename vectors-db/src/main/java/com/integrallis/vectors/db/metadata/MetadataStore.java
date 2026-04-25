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
package com.integrallis.vectors.db.metadata;

import com.integrallis.vectors.db.Document;

/**
 * Internal store mapping ordinals to full {@link Document} records.
 *
 * <p>Two implementations exist: {@link InMemoryMetadataStore} (used when no {@code storagePath} is
 * configured) and the mapped implementation in {@code com.integrallis.vectors.db.storage} (used in
 * persistent mode — the file is mmap'd via a shared per-generation {@link
 * java.lang.foreign.Arena}).
 *
 * <p><b>Thread safety.</b> Implementations are <i>not</i> required to be thread-safe on their own.
 * The facade ({@code VectorCollectionImpl}) protects a store instance by mutating it only under the
 * writer lock and publishing a fully-populated successor via a volatile {@code Generation} record —
 * readers capture one snapshot and never observe torn state.
 *
 * <p>Mapped implementations throw {@link UnsupportedOperationException} from {@link #put(int,
 * Document)} and {@link #delete(int)} — persistent generations are written whole-file via a static
 * {@code Writer.writeTo(...)} helper, not by per-entry mutation.
 */
public interface MetadataStore extends AutoCloseable {

  /** Returns the document at the given ordinal, or {@code null} if unknown. */
  Document get(int ordinal);

  /**
   * Associates the ordinal with the document, overwriting any previous mapping.
   *
   * @throws UnsupportedOperationException if this store is read-only (mapped implementations)
   */
  void put(int ordinal, Document document);

  /**
   * Removes the mapping for the given ordinal.
   *
   * @throws UnsupportedOperationException if this store is read-only (mapped implementations)
   */
  void delete(int ordinal);

  /**
   * Resolves a batch of ordinals in order. Index {@code i} of the returned array corresponds to
   * {@code ordinals[i]}.
   */
  Document[] bulkRead(int[] ordinals);

  /** Returns the number of stored documents. */
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
