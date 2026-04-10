package com.integrallis.vectors.db;

import java.util.List;
import java.util.Objects;

/**
 * Typed metadata value attached to a {@link Document}. Sealed to enforce exhaustive pattern
 * matching by the filter executor (Step 5) and the metadata store.
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
