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
import java.util.Objects;

/**
 * Predicate AST for metadata filtering. Sealed to enable exhaustive pattern matching in {@link
 * FilterExecutor}.
 *
 * <p>The sealed hierarchy allows the compiler to enforce exhaustive pattern matching over all
 * filter types. {@link FilterExecutor#matches(Filter, java.util.Map)} evaluates a filter against a
 * document's metadata; the {@link com.integrallis.vectors.db.VectorCollection} facade applies this
 * as a post-filter on ANN search candidates.
 */
public sealed interface Filter
    permits Filter.All,
        Filter.Eq,
        Filter.NumericRange,
        Filter.In,
        Filter.And,
        Filter.Or,
        Filter.Not {

  /** Accepts every document. */
  record All() implements Filter {}

  /** Equality against a typed literal. */
  record Eq(String field, Object value) implements Filter {
    public Eq {
      Objects.requireNonNull(field, "field must not be null");
      Objects.requireNonNull(value, "value must not be null");
    }
  }

  /**
   * Numeric range predicate. Bounds are inclusive when the corresponding {@code inclusive} flag is
   * {@code true}; null bounds mean unbounded.
   */
  record NumericRange(
      String field, Double lower, boolean lowerInclusive, Double upper, boolean upperInclusive)
      implements Filter {
    public NumericRange {
      Objects.requireNonNull(field, "field must not be null");
    }
  }

  /** IN predicate — membership test against a set of string or numeric values. */
  record In(String field, List<Object> values) implements Filter {
    public In {
      Objects.requireNonNull(field, "field must not be null");
      values = List.copyOf(values);
    }
  }

  /** Logical AND of child filters. Requires at least one child. */
  record And(List<Filter> children) implements Filter {
    public And {
      Objects.requireNonNull(children, "children must not be null");
      if (children.isEmpty()) {
        throw new IllegalArgumentException("And requires at least one child filter");
      }
      children = List.copyOf(children);
    }
  }

  /** Logical OR of child filters. Requires at least one child. */
  record Or(List<Filter> children) implements Filter {
    public Or {
      Objects.requireNonNull(children, "children must not be null");
      if (children.isEmpty()) {
        throw new IllegalArgumentException("Or requires at least one child filter");
      }
      children = List.copyOf(children);
    }
  }

  /** Logical NOT of a child filter. */
  record Not(Filter child) implements Filter {
    public Not {
      Objects.requireNonNull(child, "child must not be null");
    }
  }
}
