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
package com.integrallis.vectors.core;

import java.util.List;
import java.util.Objects;

/**
 * Typed metadata value attached to a {@link Document}. Sealed to enforce exhaustive pattern
 * matching by filter executors and metadata stores.
 *
 * <p>Supported variants:
 *
 * <ul>
 *   <li>{@link Str} — string field
 *   <li>{@link Num} — numeric field (double; covers both integral and floating-point)
 *   <li>{@link Bool} — boolean field
 *   <li>{@link Tags} — list of string tags (e.g., for IN filters)
 * </ul>
 */
public sealed interface MetadataValue
    permits MetadataValue.Str, MetadataValue.Num, MetadataValue.Bool, MetadataValue.Tags {

  /** String-valued metadata. */
  record Str(String value) implements MetadataValue {
    public Str {
      Objects.requireNonNull(value, "value must not be null");
    }
  }

  /** Numeric-valued metadata (double precision). */
  record Num(double value) implements MetadataValue {
    /** Convenience factory from {@code long}. */
    public static Num of(long v) {
      return new Num((double) v);
    }

    /** Convenience factory from {@code double}. */
    public static Num of(double v) {
      return new Num(v);
    }
  }

  /** Boolean-valued metadata. */
  record Bool(boolean value) implements MetadataValue {}

  /** List-of-string-tags metadata. */
  record Tags(List<String> values) implements MetadataValue {
    public Tags {
      Objects.requireNonNull(values, "values must not be null");
      values = List.copyOf(values);
    }
  }

  /** Convenience factory for string values. */
  static MetadataValue of(String v) {
    return new Str(v);
  }

  /** Convenience factory for {@code long} values. */
  static MetadataValue of(long v) {
    return Num.of(v);
  }

  /** Convenience factory for {@code double} values. */
  static MetadataValue of(double v) {
    return Num.of(v);
  }

  /** Convenience factory for boolean values. */
  static MetadataValue of(boolean v) {
    return new Bool(v);
  }

  /** Convenience factory for a tag list. */
  static MetadataValue tags(String... v) {
    return new Tags(List.of(v));
  }
}
