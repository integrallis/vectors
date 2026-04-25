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

import java.util.List;

/**
 * Static factory for {@link Filter} nodes. All filter types are evaluated by {@link
 * FilterExecutor#matches(Filter, java.util.Map)} during post-filter search in the {@link
 * com.integrallis.vectors.db.VectorCollection} facade.
 */
public final class Filters {

  private Filters() {}

  /** Returns the constant "match everything" filter. */
  public static Filter all() {
    return new Filter.All();
  }

  /** Equality predicate on a string field. */
  public static Filter eq(String field, String value) {
    return new Filter.Eq(field, value);
  }

  /** Equality predicate on an integral numeric field. */
  public static Filter eq(String field, long value) {
    return new Filter.Eq(field, value);
  }

  /** Equality predicate on a double-precision numeric field. */
  public static Filter eq(String field, double value) {
    return new Filter.Eq(field, value);
  }

  /** Equality predicate on a boolean field. */
  public static Filter eq(String field, boolean value) {
    return new Filter.Eq(field, value);
  }

  /** Greater-than-or-equal numeric predicate. */
  public static Filter gte(String field, double value) {
    return new Filter.NumericRange(field, value, true, null, false);
  }

  /** Less-than-or-equal numeric predicate. */
  public static Filter lte(String field, double value) {
    return new Filter.NumericRange(field, null, false, value, true);
  }

  /** Strict greater-than numeric predicate. */
  public static Filter gt(String field, double value) {
    return new Filter.NumericRange(field, value, false, null, false);
  }

  /** Strict less-than numeric predicate. */
  public static Filter lt(String field, double value) {
    return new Filter.NumericRange(field, null, false, value, false);
  }

  /** Inclusive {@code [lower, upper]} numeric range. */
  public static Filter between(String field, double lower, double upper) {
    return new Filter.NumericRange(field, lower, true, upper, true);
  }

  /** IN predicate over a set of string values. */
  public static Filter inStr(String field, String... values) {
    return new Filter.In(field, List.of((Object[]) values));
  }

  /** IN predicate over a set of numeric values. */
  public static Filter inNum(String field, double... values) {
    Object[] boxed = new Object[values.length];
    for (int i = 0; i < values.length; i++) {
      boxed[i] = values[i];
    }
    return new Filter.In(field, List.of(boxed));
  }

  /** Logical AND of child filters. */
  public static Filter and(Filter... children) {
    return new Filter.And(List.of(children));
  }

  /** Logical OR of child filters. */
  public static Filter or(Filter... children) {
    return new Filter.Or(List.of(children));
  }

  /** Logical NOT of a child filter. */
  public static Filter not(Filter child) {
    return new Filter.Not(child);
  }
}
