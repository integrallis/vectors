package com.integrallis.vectors.db.id;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * In-memory bidirectional mapping between external string ids and dense int ordinals. Ordinals are
 * assigned sequentially starting from 0.
 *
 * <p>Step 2 scope: only {@link #put(String)} is used by the add path. Duplicates throw because
 * {@code add} is strict; {@code upsert} is deferred to Step 3.
 */
public final class InMemoryIdMapper {

  private final Map<String, Integer> idToOrdinal = new HashMap<>();
  private final List<String> ordinalToId = new ArrayList<>();

  /**
   * Registers an external id and returns its assigned ordinal.
   *
   * @throws IllegalArgumentException if the id has already been registered
   */
  public int put(String id) {
    Objects.requireNonNull(id, "id must not be null");
    if (idToOrdinal.containsKey(id)) {
      throw new IllegalArgumentException("Duplicate id: " + id);
    }
    int ordinal = ordinalToId.size();
    ordinalToId.add(id);
    idToOrdinal.put(id, ordinal);
    return ordinal;
  }

  /** Returns {@code true} if the external id is known to this mapper. */
  public boolean contains(String id) {
    return idToOrdinal.containsKey(id);
  }

  /** Returns the ordinal assigned to the id, or {@code -1} if unknown. */
  public int ordinalOf(String id) {
    Integer ord = idToOrdinal.get(id);
    return ord == null ? -1 : ord;
  }

  /** Returns the external id for the given ordinal. */
  public String idOf(int ordinal) {
    if (ordinal < 0 || ordinal >= ordinalToId.size()) {
      throw new IndexOutOfBoundsException("ordinal out of range: " + ordinal);
    }
    return ordinalToId.get(ordinal);
  }

  /** Number of mappings currently stored. */
  public int size() {
    return ordinalToId.size();
  }

  /** Clears the mapper. */
  public void clear() {
    idToOrdinal.clear();
    ordinalToId.clear();
  }
}
