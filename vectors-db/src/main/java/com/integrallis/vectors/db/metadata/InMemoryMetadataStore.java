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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.internal.ChunkedList;
import java.util.Objects;

/**
 * In-memory {@link MetadataStore} backed by a {@link ChunkedList} of {@link Document} references,
 * indexed by ordinal.
 *
 * <p><b>Why a ChunkedList instead of a HashMap.</b> Ordinals are dense, contiguous, and append-only
 * in the commit pipeline (the next ordinal is always {@code parent.size() + delta.size()}). A
 * chunked array gives O(1) lookups and O(num_chunks) commit-time snapshot — versus the prior
 * HashMap implementation's O(N) deep copy on every commit. The bench-measured win at 100k entries
 * is ≥30× ({@code CommitPipelineBenchmark}).
 *
 * <p><b>Deletes.</b> The commit pipeline does not call {@link #delete(int)} — tombstoned ordinals
 * are marked on the {@code Generation}'s {@code BitSet}, not by removing entries here (the comment
 * in {@code VectorCollectionImpl.commitInMemory} explains why: graph traversal must never operate
 * on a zeroed vector slot). The method is supported for the interface contract by setting the
 * ordinal's slot to {@code null}, which makes subsequent {@link #get(int)} return {@code null}.
 *
 * <p><b>Sparse puts.</b> Not supported. The commit pipeline always puts at sequential ordinals.
 * Calling {@link #put(int, Document)} with {@code ordinal > size()} throws {@link
 * IllegalArgumentException}.
 *
 * <p><b>Mutable vs. immutable phase.</b> The store is held as a {@link ChunkedList.Builder} so a
 * commit's batch of {@code put}s appends in O(1) amortized. Like the id mapper, it must not be
 * mutated once its enclosing generation is published; the commit pipeline guarantees this by always
 * {@code copyOf}-ing to a fresh successor.
 *
 * <p><b>Thread safety.</b> Not thread-safe by itself. The facade ({@code VectorCollectionImpl})
 * mutates a store only under the writer lock and publishes the successor via a volatile {@code
 * Generation} record.
 */
public final class InMemoryMetadataStore implements MetadataStore {

  /**
   * Backing list: index = ordinal, value = Document (or {@code null} if deleted or never put). Held
   * as a builder so a commit's batch of appends is O(batch), not O(batch × CHUNK_SIZE).
   */
  private ChunkedList.Builder<Document> byOrdinal;

  /** Creates an empty store. */
  public InMemoryMetadataStore() {
    this.byOrdinal = ChunkedList.<Document>empty().toBuilder();
  }

  /**
   * Creates a successor sharing structure with {@code other}. The reverse builder is rebuilt into
   * an immutable snapshot and a fresh builder derived from it — the successor shares {@code
   * other}'s full chunks by reference and owns a freshly-cloned tail chunk. O(N / CHUNK_SIZE).
   * Subsequent puts on the successor append to its own builder; {@code other} is unaffected.
   */
  public static InMemoryMetadataStore copyOf(InMemoryMetadataStore other) {
    Objects.requireNonNull(other, "other must not be null");
    InMemoryMetadataStore copy = new InMemoryMetadataStore();
    copy.byOrdinal = other.byOrdinal.build().toBuilder();
    return copy;
  }

  @Override
  public Document get(int ordinal) {
    if (ordinal < 0 || ordinal >= byOrdinal.size()) {
      return null;
    }
    return byOrdinal.get(ordinal);
  }

  /**
   * Associates {@code ordinal} with {@code document}.
   *
   * @throws IllegalArgumentException if {@code ordinal > size()} — sparse puts that would leave a
   *     gap in the dense ordinal range are not supported. Callers must add documents at sequential
   *     ordinals (the commit pipeline already does this).
   * @throws NullPointerException if {@code document} is null
   */
  @Override
  public void put(int ordinal, Document document) {
    Objects.requireNonNull(document, "document must not be null");
    int current = byOrdinal.size();
    if (ordinal == current) {
      byOrdinal.append(document);
    } else if (ordinal >= 0 && ordinal < current) {
      byOrdinal.set(ordinal, document);
    } else {
      throw new IllegalArgumentException(
          "Sparse put not supported: ordinal "
              + ordinal
              + " > current size "
              + current
              + ". Callers must put at sequential ordinals.");
    }
  }

  /**
   * Clears the mapping at {@code ordinal} (subsequent {@link #get(int)} returns null). Not used by
   * the commit pipeline (which tombstones via the {@code Generation} BitSet) — supported for the
   * interface contract and incidental callers.
   */
  @Override
  public void delete(int ordinal) {
    if (ordinal < 0 || ordinal >= byOrdinal.size()) {
      return;
    }
    byOrdinal.set(ordinal, null);
  }

  @Override
  public Document[] bulkRead(int[] ordinals) {
    Objects.requireNonNull(ordinals, "ordinals must not be null");
    Document[] out = new Document[ordinals.length];
    int sz = byOrdinal.size();
    for (int i = 0; i < ordinals.length; i++) {
      int ord = ordinals[i];
      out[i] = (ord < 0 || ord >= sz) ? null : byOrdinal.get(ord);
    }
    return out;
  }

  @Override
  public int size() {
    return byOrdinal.size();
  }

  /** Clears the store back to empty. */
  public void clear() {
    byOrdinal = ChunkedList.<Document>empty().toBuilder();
  }
}
