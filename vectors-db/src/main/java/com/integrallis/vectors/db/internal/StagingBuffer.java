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
package com.integrallis.vectors.db.internal;

import com.integrallis.vectors.core.Document;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Mutable per-collection staging buffer that accumulates {@link Document}s and tombstone intents
 * written since the last commit. A {@link LinkedHashMap} serves as the primary store, providing
 * O(1) lookup ({@link #contains}), O(1) removal ({@link #removeDocument}), and preserving insertion
 * order for the commit pipeline's ordinal-assignment loop.
 *
 * <p><b>Thread safety.</b> {@code StagingBuffer} is <i>not</i> thread-safe on its own. The facade
 * ({@code VectorCollectionImpl}) guards all mutation through a single {@code ReentrantLock} (writer
 * lock). This type deliberately inherits safety from the enclosing lock so that reads of the
 * published {@code Generation} record can stay lock-free.
 *
 * <p>Holds both newly-added documents and pending tombstone ids. Tombstones are ids that should be
 * marked as deleted in the next committed generation. The {@link #hasWork()} method returns {@code
 * true} when either documents or tombstones are pending, ensuring tombstone-only commits produce a
 * new generation.
 */
public final class StagingBuffer {

  /**
   * Primary document store. {@link LinkedHashMap} preserves insertion order (needed by the commit
   * pipeline's sequential ordinal assignment) while providing O(1) {@link #contains} and O(1)
   * {@link #removeDocument} — replacing the previous {@code ArrayList} + shadow {@code
   * LinkedHashSet} pair whose {@code removeIf} scan was O(n).
   */
  private final LinkedHashMap<String, Document> documentsMap = new LinkedHashMap<>();

  private final Set<String> pendingTombstones = new LinkedHashSet<>();

  /**
   * Appends a document if its id is not already staged. Returns {@code true} on success, {@code
   * false} if the id was already present in the staging map.
   *
   * <p><b>Defensive copy of the vector.</b> The caller-supplied {@code float[]} is cloned at the
   * staging boundary so that the collection's internal copy is decoupled from any reference the
   * caller still holds. This pays the vector-clone cost <i>once</i> per insert, instead of paying
   * it for every committed vector on every subsequent {@code commit()} (which is what the prior
   * model did). Combined with shared-by-reference vectors in {@code VectorCollectionImpl.commit*},
   * this makes the in-memory commit path O(staged) in allocations rather than O(live).
   *
   * <p>Metadata and other Document fields are immutable (the Document record's canonical
   * constructor already defensively copies the metadata map), so cloning only the vector is
   * sufficient to fully isolate the stored Document from caller-side mutation.
   */
  public boolean append(Document doc) {
    Objects.requireNonNull(doc, "doc must not be null");
    if (documentsMap.containsKey(doc.id())) {
      return false;
    }
    Document defensive =
        doc.vector() == null
            ? doc
            : new Document(doc.id(), doc.vector().clone(), doc.text(), doc.metadata());
    documentsMap.put(doc.id(), defensive);
    return true;
  }

  /** Returns {@code true} if {@code id} is currently staged as an add. */
  public boolean contains(String id) {
    return documentsMap.containsKey(id);
  }

  /** Number of documents staged for add since the last commit. */
  public int size() {
    return documentsMap.size();
  }

  /** Returns {@code true} if no documents are staged for add. */
  public boolean isEmpty() {
    return documentsMap.isEmpty();
  }

  /**
   * Returns {@code true} if there is any pending work — either staged documents or pending
   * tombstones. Used by the commit path to decide whether a new generation must be produced.
   */
  public boolean hasWork() {
    return !documentsMap.isEmpty() || !pendingTombstones.isEmpty();
  }

  /**
   * Returns a snapshot {@link List} of the staged documents in insertion order. The list is a fresh
   * copy: mutations to it do not affect the buffer's internal state, and subsequent calls to {@link
   * #append}/{@link #removeDocument}/{@link #clear} do not affect previously returned snapshots.
   * Callers may use indexed access ({@code list.get(i)}) safely.
   *
   * <p>The caller must hold the enclosing writer lock before calling this method.
   */
  public List<Document> documents() {
    return new ArrayList<>(documentsMap.values());
  }

  /**
   * Records a tombstone intent for the given id. Returns {@code true} if the id was newly added to
   * the pending tombstones set, {@code false} if it was already pending.
   */
  public boolean stageDelete(String id) {
    Objects.requireNonNull(id, "id must not be null");
    return pendingTombstones.add(id);
  }

  /** Returns {@code true} if the given id is pending deletion. */
  public boolean isTombstoned(String id) {
    return pendingTombstones.contains(id);
  }

  /** Returns an unmodifiable view of the pending tombstone ids. */
  public Set<String> pendingTombstones() {
    return Collections.unmodifiableSet(pendingTombstones);
  }

  /**
   * Removes a staged document by id in O(1). Used by {@code delete()} to undo a staged add (delete
   * before commit) and by {@code upsert()} to replace a staged document. Returns {@code true} if
   * the document was found and removed, {@code false} if the id was not staged.
   */
  public boolean removeDocument(String id) {
    Objects.requireNonNull(id, "id must not be null");
    return documentsMap.remove(id) != null;
  }

  /** Drops all staged documents and pending tombstones. */
  public void clear() {
    documentsMap.clear();
    pendingTombstones.clear();
  }
}
