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
package com.integrallis.vectors.optimizer.space;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * One axis of a {@link SearchSpace}. Sealed: a parameter is either categorical, an integer or
 * double range (optionally log-scaled), a discrete enumeration of arbitrary values, or a fixed
 * constant carried for trial-record completeness.
 *
 * @param <T> Java type produced when this axis is sampled
 */
public sealed interface ParamSpec<T>
    permits ParamSpec.Categorical,
        ParamSpec.IntRange,
        ParamSpec.DoubleRange,
        ParamSpec.Discrete,
        ParamSpec.FixedString,
        ParamSpec.FixedInt,
        ParamSpec.FixedDouble {

  /** Axis name; unique within a {@link SearchSpace}. */
  String name();

  /** Cardinality of this axis. {@link OptionalLong#empty()} for continuous (DoubleRange) axes. */
  OptionalLong cardinality();

  /** Categorical axis with a fixed list of string values. */
  record Categorical(String name, List<String> values) implements ParamSpec<String> {
    public Categorical {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(values, "values");
      if (values.isEmpty()) {
        throw new IllegalArgumentException("Categorical " + name + " requires at least one value");
      }
      values = List.copyOf(values);
    }

    @Override
    public OptionalLong cardinality() {
      return OptionalLong.of(values.size());
    }
  }

  /** Integer range [min, max] inclusive on both ends. */
  record IntRange(String name, int min, int max, boolean logScale) implements ParamSpec<Integer> {
    public IntRange {
      Objects.requireNonNull(name, "name");
      if (min > max) {
        throw new IllegalArgumentException(
            "IntRange " + name + ": min (" + min + ") must be <= max (" + max + ")");
      }
      if (logScale && min <= 0) {
        throw new IllegalArgumentException(
            "IntRange " + name + ": logScale requires min > 0 (got " + min + ")");
      }
    }

    @Override
    public OptionalLong cardinality() {
      return OptionalLong.of((long) max - (long) min + 1L);
    }
  }

  /** Double range [min, max] inclusive on both ends. Continuous: cardinality is empty. */
  record DoubleRange(String name, double min, double max, boolean logScale)
      implements ParamSpec<Double> {
    public DoubleRange {
      Objects.requireNonNull(name, "name");
      if (Double.isNaN(min) || Double.isNaN(max)) {
        throw new IllegalArgumentException("DoubleRange " + name + ": NaN bounds");
      }
      if (min > max) {
        throw new IllegalArgumentException(
            "DoubleRange " + name + ": min (" + min + ") must be <= max (" + max + ")");
      }
      if (logScale && min <= 0.0) {
        throw new IllegalArgumentException(
            "DoubleRange " + name + ": logScale requires min > 0 (got " + min + ")");
      }
    }

    @Override
    public OptionalLong cardinality() {
      return OptionalLong.empty();
    }
  }

  /** Discrete set of typed values (e.g. enum constants, integer codebook sizes). */
  record Discrete<T>(String name, List<T> values) implements ParamSpec<T> {
    public Discrete {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(values, "values");
      if (values.isEmpty()) {
        throw new IllegalArgumentException("Discrete " + name + " requires at least one value");
      }
      values = List.copyOf(values);
    }

    @Override
    public OptionalLong cardinality() {
      return OptionalLong.of(values.size());
    }
  }

  /** Fixed string constant; cardinality 1. Useful for documenting a pinned axis. */
  record FixedString(String name, String value) implements ParamSpec<String> {
    public FixedString {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(value, "value");
    }

    @Override
    public OptionalLong cardinality() {
      return OptionalLong.of(1L);
    }
  }

  /** Fixed integer constant; cardinality 1. */
  record FixedInt(String name, int value) implements ParamSpec<Integer> {
    public FixedInt {
      Objects.requireNonNull(name, "name");
    }

    @Override
    public OptionalLong cardinality() {
      return OptionalLong.of(1L);
    }
  }

  /** Fixed double constant; cardinality 1. */
  record FixedDouble(String name, double value) implements ParamSpec<Double> {
    public FixedDouble {
      Objects.requireNonNull(name, "name");
    }

    @Override
    public OptionalLong cardinality() {
      return OptionalLong.of(1L);
    }
  }
}
