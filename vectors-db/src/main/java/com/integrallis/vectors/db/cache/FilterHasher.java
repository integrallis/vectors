package com.integrallis.vectors.db.cache;

import com.integrallis.vectors.db.filter.Filter;
import java.util.List;
import java.util.Objects;

/**
 * Computes a stable {@code long} hash for a {@link Filter} expression.
 *
 * <p>The hash is used as part of the {@link QvCacheKey} to distinguish queries with different
 * filter predicates. Semantically equivalent filters that happen to be structurally identical
 * produce the same hash; structurally different but semantically equivalent filters (e.g.,
 * reordered {@code And} children) may produce different hashes — this is acceptable: it results in
 * a cache miss (never a stale result).
 *
 * <p>All permitted {@link Filter} subtypes are handled exhaustively via pattern matching on the
 * sealed interface.
 */
public final class FilterHasher {

  private FilterHasher() {}

  /**
   * Returns a {@code long} hash for {@code filter}.
   *
   * @param filter the filter to hash; {@code null} is treated as {@link Filter.All} (no filter)
   * @return a stable hash code for the filter expression
   */
  public static long hash(Filter filter) {
    if (filter == null) return 0L;
    return switch (filter) {
      case Filter.All ignored -> 0L;
      case Filter.Eq eq -> mix(1, Objects.hash(eq.field(), eq.value()));
      case Filter.NumericRange nr ->
          mix(
              2,
              Objects.hash(
                  nr.field(), nr.lower(), nr.lowerInclusive(), nr.upper(), nr.upperInclusive()));
      case Filter.In in -> mix(3, Objects.hash(in.field(), in.values()));
      case Filter.And and -> mix(4, hashList(and.children()));
      case Filter.Or or -> mix(5, hashList(or.children()));
      case Filter.Not not -> mix(6, (int) hash(not.child()));
    };
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private static long mix(int typeTag, int inner) {
    // Combine type tag + inner hash via FNV-inspired mixing to reduce collisions.
    long h = 0xcbf29ce484222325L;
    h ^= typeTag;
    h *= 0x100000001b3L;
    h ^= (inner & 0xFFFFFFFFL);
    h *= 0x100000001b3L;
    return h;
  }

  private static int hashList(List<Filter> children) {
    int h = 1;
    for (Filter child : children) {
      h = 31 * h + (int) hash(child);
    }
    return h;
  }
}
