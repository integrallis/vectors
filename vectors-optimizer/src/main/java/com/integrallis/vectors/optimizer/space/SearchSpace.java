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

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;

/**
 * An ordered list of {@link ParamSpec} axes that defines the joint search space for a study.
 *
 * <p>Axis names must be unique. The total cardinality is the product of per-axis cardinalities; if
 * any axis is continuous (e.g. {@link ParamSpec.DoubleRange}) the total is reported as empty
 * (infinite).
 */
public record SearchSpace(List<ParamSpec<?>> axes) {

  public SearchSpace {
    Objects.requireNonNull(axes, "axes");
    Set<String> seen = new HashSet<>();
    for (ParamSpec<?> a : axes) {
      Objects.requireNonNull(a, "axis");
      if (!seen.add(a.name())) {
        throw new IllegalArgumentException("Duplicate axis name: " + a.name());
      }
    }
    axes = List.copyOf(axes);
  }

  /** Look up an axis by name. */
  public Optional<ParamSpec<?>> byName(String name) {
    for (ParamSpec<?> a : axes) {
      if (a.name().equals(name)) {
        return Optional.of(a);
      }
    }
    return Optional.empty();
  }

  /**
   * Joint cardinality of the space, or {@link OptionalLong#empty()} when any axis is continuous.
   *
   * <p>The product is computed in {@code long} arithmetic with overflow detection; on overflow
   * (i.e. > {@link Long#MAX_VALUE}) we also return empty.
   */
  public OptionalLong cardinalityOrInfinite() {
    long product = 1L;
    for (ParamSpec<?> a : axes) {
      OptionalLong c = a.cardinality();
      if (c.isEmpty()) {
        return OptionalLong.empty();
      }
      try {
        product = Math.multiplyExact(product, c.getAsLong());
      } catch (ArithmeticException overflow) {
        return OptionalLong.empty();
      }
    }
    return OptionalLong.of(product);
  }
}
