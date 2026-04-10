package com.integrallis.vectors.db.metadata;

import com.integrallis.vectors.db.Document;

/**
 * Internal store mapping ordinals to full {@link Document} records. Step 2 is in-memory and
 * per-collection; persistence lands in Step 4.
 */
public interface MetadataStore {

  /** Returns the document at the given ordinal, or {@code null} if unknown. */
  Document get(int ordinal);

  /** Associates the ordinal with the document, overwriting any previous mapping. */
  void put(int ordinal, Document document);

  /** Removes the mapping for the given ordinal. */
  void delete(int ordinal);

  /**
   * Resolves a batch of ordinals in order. Index {@code i} of the returned array corresponds to
   * {@code ordinals[i]}.
   */
  Document[] bulkRead(int[] ordinals);

  /** Returns the number of stored documents. */
  int size();
}
