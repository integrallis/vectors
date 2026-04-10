package com.integrallis.vectors.db.metadata;

import com.integrallis.vectors.db.Document;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/** In-memory {@link MetadataStore} backed by a {@link HashMap}. */
public final class InMemoryMetadataStore implements MetadataStore {

  private final Map<Integer, Document> byOrdinal = new HashMap<>();

  @Override
  public Document get(int ordinal) {
    return byOrdinal.get(ordinal);
  }

  @Override
  public void put(int ordinal, Document document) {
    Objects.requireNonNull(document, "document must not be null");
    byOrdinal.put(ordinal, document);
  }

  @Override
  public void delete(int ordinal) {
    byOrdinal.remove(ordinal);
  }

  @Override
  public Document[] bulkRead(int[] ordinals) {
    Objects.requireNonNull(ordinals, "ordinals must not be null");
    Document[] out = new Document[ordinals.length];
    for (int i = 0; i < ordinals.length; i++) {
      out[i] = byOrdinal.get(ordinals[i]);
    }
    return out;
  }

  @Override
  public int size() {
    return byOrdinal.size();
  }

  /** Clears the store. */
  public void clear() {
    byOrdinal.clear();
  }
}
