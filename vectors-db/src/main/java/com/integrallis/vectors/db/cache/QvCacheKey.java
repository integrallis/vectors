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
package com.integrallis.vectors.db.cache;

import java.util.Arrays;

/**
 * Immutable cache key for {@link QvCache} entries.
 *
 * <p>Three components contribute to key equality:
 *
 * <ol>
 *   <li>{@link #quantizedQuery} — the query vector quantized to signed int8 (one byte per
 *       dimension) using max-absolute-value normalization. Nearby queries that differ only in the
 *       least-significant bits of each dimension map to the same key, giving the cache its
 *       "semantic" grouping property.
 *   <li>{@link #k} — number of results requested.
 *   <li>{@link #filterHash} — stable hash of the filter expression, as computed by {@link
 *       FilterHasher#hash(com.integrallis.vectors.db.filter.Filter)}.
 * </ol>
 *
 * <p>{@code equals} and {@code hashCode} use {@link Arrays#equals} / {@link Arrays#hashCode} for
 * the byte array so that structurally identical quantized queries always compare equal.
 */
public final class QvCacheKey {

  private final byte[] quantizedQuery;
  private final int k;
  private final long filterHash;
  private final int cachedHashCode;

  /**
   * Constructs a cache key.
   *
   * @param quantizedQuery int8-quantized query vector (one byte per dimension); not null; will be
   *     stored by reference — callers must not mutate after passing
   * @param k number of requested results
   * @param filterHash hash produced by {@link FilterHasher}
   */
  public QvCacheKey(byte[] quantizedQuery, int k, long filterHash) {
    this.quantizedQuery = quantizedQuery;
    this.k = k;
    this.filterHash = filterHash;
    this.cachedHashCode = computeHashCode();
  }

  // ---------------------------------------------------------------------------
  // Accessors
  // ---------------------------------------------------------------------------

  public byte[] quantizedQuery() {
    return quantizedQuery;
  }

  public int k() {
    return k;
  }

  public long filterHash() {
    return filterHash;
  }

  // ---------------------------------------------------------------------------
  // equals / hashCode — Arrays.equals for byte array
  // ---------------------------------------------------------------------------

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof QvCacheKey other)) return false;
    return k == other.k
        && filterHash == other.filterHash
        && Arrays.equals(quantizedQuery, other.quantizedQuery);
  }

  @Override
  public int hashCode() {
    return cachedHashCode;
  }

  private int computeHashCode() {
    int result = Arrays.hashCode(quantizedQuery);
    result = 31 * result + k;
    result = 31 * result + Long.hashCode(filterHash);
    return result;
  }

  @Override
  public String toString() {
    return "QvCacheKey{k="
        + k
        + ", filterHash="
        + filterHash
        + ", dim="
        + quantizedQuery.length
        + "}";
  }
}
