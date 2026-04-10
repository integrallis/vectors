package com.integrallis.vectors.db.internal;

import com.integrallis.vectors.db.Document;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Mutable per-collection staging buffer that accumulates {@link Document}s written since the last
 * commit. A shadow {@code Set<String>} provides O(1) duplicate-id detection alongside the ordered
 * {@code List<Document>} used to materialize a committed generation.
 *
 * <p><b>Thread safety.</b> {@code StagingBuffer} is <i>not</i> thread-safe on its own. The facade
 * ({@code VectorCollectionImpl}) guards all mutation through a single {@code ReentrantLock} (writer
 * lock). This type deliberately inherits safety from the enclosing lock so that reads of the
 * published {@code Generation} record can stay lock-free.
 *
 * <p>Step 3 holds only newly-added documents. Pending tombstones and upserts will live here in Step
 * 6.
 */
public final class StagingBuffer {

  private final List<Document> documents = new ArrayList<>();
  private final Set<String> ids = new HashSet<>();

  /**
   * Appends a document if its id is not already staged. Returns {@code true} on success, {@code
   * false} if the id was already present in the staging set.
   */
  public boolean append(Document doc) {
    Objects.requireNonNull(doc, "doc must not be null");
    if (!ids.add(doc.id())) {
      return false;
    }
    documents.add(doc);
    return true;
  }

  /** Returns {@code true} if {@code id} is currently staged. */
  public boolean contains(String id) {
    return ids.contains(id);
  }

  /** Number of documents staged since the last commit. */
  public int size() {
    return documents.size();
  }

  /** Returns {@code true} if no documents are staged. */
  public boolean isEmpty() {
    return documents.isEmpty();
  }

  /**
   * Returns the live document list. The caller must hold the enclosing writer lock and must not
   * mutate the returned list outside this type's own methods.
   */
  public List<Document> documents() {
    return documents;
  }

  /** Drops all staged documents and their shadow ids. */
  public void clear() {
    documents.clear();
    ids.clear();
  }
}
