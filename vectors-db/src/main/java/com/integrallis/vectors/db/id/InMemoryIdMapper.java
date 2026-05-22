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

import com.integrallis.vectors.db.internal.ChunkedList;
import com.integrallis.vectors.db.internal.Hamt;
import java.util.Objects;

/**
 * In-memory {@link IdMapper} backed by structurally-sharing persistent collections.
 *
 * <p><b>Layout.</b> Two collections cover the bidirectional mapping:
 *
 * <ul>
 *   <li>{@link Hamt} for the forward map {@code id -> ordinal} (sparse, hash-keyed)
 *   <li>{@link ChunkedList} for the reverse map {@code ordinal -> id} (dense, integer-keyed,
 *       append-only)
 * </ul>
 *
 * <p><b>Commit cost.</b> {@link #copyOf(InMemoryIdMapper)} is now O(1) in shared-state (Hamt root
 * reference + ChunkedList outer-array reference), versus the previous O(N) {@code HashMap.putAll +
 * ArrayList.addAll}. For a collection with 100k live documents that's the difference between ~10 ms
 * per commit and {@code <1 μs} — measured by {@code CommitPipelineBenchmark}.
 *
 * <p><b>Read cost.</b> {@link #ordinalOf(String)} is O(log₃₂ N) ≈ 120 ns (Hamt walk). {@link
 * #idOf(int)} is O(1) ≈ 10 ns (chunk-list index). {@link #contains(String)} is O(log₃₂ N).
 *
 * <p><b>Mutable vs. immutable phase.</b> Like the prior implementation, a mapper has a mutable
 * phase (during commit, single-threaded under the facade's writer lock) and an immutable phase
 * (after the enclosing {@code Generation} is published via a volatile write). The reverse map is
 * held as a {@link ChunkedList.Builder} so the mutable phase can append in O(1) amortized without
 * the per-append tail-chunk clone the one-shot {@link ChunkedList#append} pays. The builder must
 * not be mutated once its mapper's generation is published; the commit pipeline guarantees this by
 * always {@code copyOf}-ing to a fresh successor mapper.
 *
 * <p><b>Not thread-safe by itself.</b> The facade ({@code VectorCollectionImpl}) protects a mapper
 * instance by holding the writer lock during mutation and publishing the fully-populated successor
 * via a {@code volatile Generation} record — the JMM happens-before contract on the publish makes
 * all preceding writes to {@code forward}/{@code reverse} visible to readers.
 */
public final class InMemoryIdMapper implements IdMapper {

  /** Forward {@code id -> ordinal} map. Replaced (not mutated) on every put. */
  private Hamt<String, Integer> forward;

  /**
   * Reverse {@code ordinal -> id} list, held as a builder so a batch of {@code put}s in one commit
   * appends in O(1) amortized. Frozen by convention once the enclosing generation is published.
   */
  private ChunkedList.Builder<String> reverse;

  /** Creates an empty mapper. */
  public InMemoryIdMapper() {
    this.forward = Hamt.empty();
    this.reverse = ChunkedList.<String>empty().toBuilder();
  }

  /**
   * Creates a successor sharing structure with {@code other}. The forward map is inherited by
   * reference (Hamt is immutable). The reverse builder is rebuilt into an immutable snapshot and a
   * fresh builder is derived from it — the successor's reverse builder shares {@code other}'s full
   * chunks by reference and owns a freshly-cloned tail chunk, so neither side's appends disturb the
   * other. O(num_chunks + CHUNK_SIZE) ≈ O(N / 1024) — for N=100k, ~1 μs.
   *
   * <p><b>Independence.</b> Subsequent mutations on the successor reassign {@code forward} and
   * append to its own {@code reverse} builder; the predecessor is unaffected.
   */
  public static InMemoryIdMapper copyOf(InMemoryIdMapper other) {
    Objects.requireNonNull(other, "other must not be null");
    InMemoryIdMapper copy = new InMemoryIdMapper();
    copy.forward = other.forward;
    copy.reverse = other.reverse.build().toBuilder();
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
    if (forward.containsKey(id)) {
      throw new IllegalArgumentException("Duplicate id: " + id);
    }
    int ordinal = reverse.size();
    reverse.append(id);
    forward = forward.put(id, ordinal);
    return ordinal;
  }

  /**
   * Registers an external id at a new ordinal, replacing any existing forward mapping. The old
   * ordinal's reverse entry remains in {@link #reverse} — the caller is expected to have tombstoned
   * it so it is never returned by the search path.
   *
   * <p>Used by the commit pipeline when an upserted document collides with a tombstoned ordinal
   * from the predecessor generation.
   */
  public int putOrReplace(String id) {
    Objects.requireNonNull(id, "id must not be null");
    int ordinal = reverse.size();
    reverse.append(id);
    forward = forward.put(id, ordinal); // Hamt.put naturally overwrites the existing key
    return ordinal;
  }

  @Override
  public boolean contains(String id) {
    return forward.containsKey(id);
  }

  @Override
  public int ordinalOf(String id) {
    Integer ord = forward.get(id);
    return ord == null ? -1 : ord;
  }

  @Override
  public String idOf(int ordinal) {
    if (ordinal < 0 || ordinal >= reverse.size()) {
      throw new IndexOutOfBoundsException("ordinal out of range: " + ordinal);
    }
    return reverse.get(ordinal);
  }

  @Override
  public int size() {
    return reverse.size();
  }
}
