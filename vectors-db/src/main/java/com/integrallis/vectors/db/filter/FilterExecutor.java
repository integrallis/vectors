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
package com.integrallis.vectors.db.filter;

import com.integrallis.vectors.core.MetadataValue;
import java.util.Map;

/**
 * Evaluates a {@link Filter} predicate against a document's metadata.
 *
 * <p>This class is stateless; all methods are static. The switch over the sealed {@link Filter}
 * hierarchy is exhaustive, as is the switch over {@link MetadataValue} subtypes where applicable.
 *
 * <p><b>Missing-field semantics.</b> If a filter references a metadata field that does not exist in
 * the document's metadata, the filter evaluates to {@code false}. This is consistent with SQL
 * NULL-comparison semantics: {@code NULL = 'foo'} is false, {@code NULL > 5} is false, etc. The
 * exception is {@link Filter.Not}: {@code NOT(missing)} evaluates to {@code true} because the
 * negation of "not matched" is "matched."
 *
 * <p><b>Type-mismatch semantics.</b> If the metadata value type does not match the filter
 * expectation (e.g., a {@link Filter.NumericRange} applied to a {@link MetadataValue.Str}), the
 * filter evaluates to {@code false}. No exception is thrown — the document simply does not match.
 */
public final class FilterExecutor {

  private FilterExecutor() {}

  /**
   * Tests whether a document's metadata satisfies the given filter.
   *
   * @param filter the filter predicate (must not be null)
   * @param metadata the document's metadata map (must not be null; may be empty)
   * @return {@code true} if the metadata satisfies the filter
   */
  public static boolean matches(Filter filter, Map<String, MetadataValue> metadata) {
    if (metadata == null) {
      return filter instanceof Filter.All;
    }
    return switch (filter) {
      case Filter.All ignored -> true;
      case Filter.Eq eq -> matchesEq(eq, metadata);
      case Filter.NumericRange range -> matchesNumericRange(range, metadata);
      case Filter.In in -> matchesIn(in, metadata);
      case Filter.And and -> {
        for (Filter child : and.children()) {
          if (!matches(child, metadata)) {
            yield false;
          }
        }
        yield true;
      }
      case Filter.Or or -> {
        for (Filter child : or.children()) {
          if (matches(child, metadata)) {
            yield true;
          }
        }
        yield false;
      }
      case Filter.Not not -> !matches(not.child(), metadata);
    };
  }

  private static boolean matchesEq(Filter.Eq eq, Map<String, MetadataValue> metadata) {
    MetadataValue mv = metadata.get(eq.field());
    if (mv == null) {
      return false;
    }
    Object filterValue = eq.value();
    return switch (mv) {
      case MetadataValue.Str s -> filterValue instanceof String fs && s.value().equals(fs);
      case MetadataValue.Num n ->
          filterValue instanceof Number fn && Double.compare(n.value(), fn.doubleValue()) == 0;
      case MetadataValue.Bool b -> filterValue instanceof Boolean fb && b.value() == fb;
      case MetadataValue.Tags t -> filterValue instanceof String fs && t.values().contains(fs);
    };
  }

  private static boolean matchesNumericRange(
      Filter.NumericRange range, Map<String, MetadataValue> metadata) {
    MetadataValue mv = metadata.get(range.field());
    if (mv == null) {
      return false;
    }
    if (!(mv instanceof MetadataValue.Num num)) {
      return false;
    }
    double value = num.value();
    if (range.lower() != null) {
      int cmp = Double.compare(value, range.lower());
      if (range.lowerInclusive() ? cmp < 0 : cmp <= 0) {
        return false;
      }
    }
    if (range.upper() != null) {
      int cmp = Double.compare(value, range.upper());
      if (range.upperInclusive() ? cmp > 0 : cmp >= 0) {
        return false;
      }
    }
    return true;
  }

  private static boolean matchesIn(Filter.In in, Map<String, MetadataValue> metadata) {
    MetadataValue mv = metadata.get(in.field());
    if (mv == null) {
      return false;
    }
    return switch (mv) {
      case MetadataValue.Str s -> in.values().contains(s.value());
      case MetadataValue.Num n -> {
        double d = n.value();
        for (Object v : in.values()) {
          if (v instanceof Number num && Double.compare(d, num.doubleValue()) == 0) {
            yield true;
          }
        }
        yield false;
      }
      case MetadataValue.Bool b -> in.values().contains(b.value());
      case MetadataValue.Tags t -> {
        // Tags match IN if any tag value is in the IN set
        for (String tag : t.values()) {
          if (in.values().contains(tag)) {
            yield true;
          }
        }
        yield false;
      }
    };
  }
}
