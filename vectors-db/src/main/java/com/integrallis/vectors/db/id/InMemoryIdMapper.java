package com.integrallis.vectors.db.id;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * In-memory {@link IdMapper} backed by a {@link HashMap} and an {@link ArrayList}.
 *
 * <p>Only {@link #put(String)} is used by the add path. Duplicates throw because {@code add} is
 * strict; {@code upsert} is deferred to Step 6.
 *
 * <p>Not thread-safe on its own. The facade ({@code VectorCollectionImpl}) protects a mapper
 * instance by holding the writer lock during mutation and publishing a fresh, fully-populated
 * mapper via a volatile {@code Generation} record for readers.
 */
public final class InMemoryIdMapper implements IdMapper {

  private final Map<String, Integer> idToOrdinal;
  private final List<String> ordinalToId;

  /** Creates an empty mapper. */
  public InMemoryIdMapper() {
    this.idToOrdinal = new HashMap<>();
    this.ordinalToId = new ArrayList<>();
  }

  /**
   * Copy constructor used by the commit pipeline to produce a mutable successor generation without
   * touching the predecessor. The returned mapper is fully independent of {@code other}.
   *
   * <p><b>Why this takes the concrete type.</b> The commit-time copy is specific to the in-memory
   * shape because it needs direct access to the backing {@code HashMap} and {@code ArrayList} for
   * O(1) bulk copy. The mapped implementation landing in Step 4a is whole-file persistent
   * (materialized via a static {@code Writer.writeTo(...)}), so the commit path will branch on the
   * current mapper's type rather than relying on a shared {@code copyOf} contract on the {@link
   * IdMapper} interface.
   */
  public static InMemoryIdMapper copyOf(InMemoryIdMapper other) {
    Objects.requireNonNull(other, "other must not be null");
    InMemoryIdMapper copy = new InMemoryIdMapper();
    copy.idToOrdinal.putAll(other.idToOrdinal);
    copy.ordinalToId.addAll(other.ordinalToId);
    return copy;
  }

  /**
   * Registers an external id and returns its assigned ordinal.
   *
   * @throws IllegalArgumentException if the id has already been registered
   */
  @Override
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

  /**
   * Registers an external id at a new ordinal, replacing any existing forward mapping. The old
   * ordinal's reverse entry ({@code ordinalToId[oldOrd]}) is <em>not</em> removed — the caller is
   * expected to have tombstoned it so that it's never consulted via the forward path. Returns the
   * newly assigned ordinal.
   *
   * <p>Used by the commit pipeline when an upserted document has the same id as a tombstoned
   * ordinal in the predecessor generation.
   */
  public int putOrReplace(String id) {
    Objects.requireNonNull(id, "id must not be null");
    int ordinal = ordinalToId.size();
    ordinalToId.add(id);
    idToOrdinal.put(id, ordinal); // overwrites old mapping if present
    return ordinal;
  }

  /** Returns {@code true} if the external id is known to this mapper. */
  @Override
  public boolean contains(String id) {
    return idToOrdinal.containsKey(id);
  }

  /** Returns the ordinal assigned to the id, or {@code -1} if unknown. */
  @Override
  public int ordinalOf(String id) {
    Integer ord = idToOrdinal.get(id);
    return ord == null ? -1 : ord;
  }

  /** Returns the external id for the given ordinal. */
  @Override
  public String idOf(int ordinal) {
    if (ordinal < 0 || ordinal >= ordinalToId.size()) {
      throw new IndexOutOfBoundsException("ordinal out of range: " + ordinal);
    }
    return ordinalToId.get(ordinal);
  }

  /** Number of mappings currently stored. */
  @Override
  public int size() {
    return ordinalToId.size();
  }
}
