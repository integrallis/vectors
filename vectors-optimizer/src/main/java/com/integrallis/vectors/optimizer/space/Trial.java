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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A concrete sample drawn from a {@link SearchSpace}: a unique trial id plus the resolved value for
 * every axis. Values are stored as {@code Object} keyed by axis name; typed accessors below cast
 * back to the expected primitive / string type.
 */
public record Trial(String trialId, Map<String, Object> params) {

  public Trial {
    Objects.requireNonNull(trialId, "trialId");
    Objects.requireNonNull(params, "params");
    params = Map.copyOf(new LinkedHashMap<>(params));
  }

  /** Reads an integer-valued axis. Throws {@link IllegalStateException} if absent or wrong type. */
  public int getInt(String name) {
    Object v = require(name);
    if (v instanceof Integer i) return i;
    if (v instanceof Long l) return Math.toIntExact(l);
    throw new IllegalStateException("Axis " + name + " is not an int (was " + v.getClass() + ")");
  }

  /** Reads a double-valued axis. Throws if absent or wrong type. */
  public double getDouble(String name) {
    Object v = require(name);
    if (v instanceof Double d) return d;
    if (v instanceof Float f) return f;
    if (v instanceof Number n) return n.doubleValue();
    throw new IllegalStateException("Axis " + name + " is not a double (was " + v.getClass() + ")");
  }

  /** Reads a string-valued axis. Throws if absent or wrong type. */
  public String getString(String name) {
    Object v = require(name);
    if (v instanceof String s) return s;
    throw new IllegalStateException("Axis " + name + " is not a string (was " + v.getClass() + ")");
  }

  /** Reads an enum-valued axis stored as the {@link Enum#name()}. */
  public <E extends Enum<E>> E getEnum(String name, Class<E> enumClass) {
    Object v = require(name);
    if (enumClass.isInstance(v)) {
      return enumClass.cast(v);
    }
    if (v instanceof String s) {
      return Enum.valueOf(enumClass, s);
    }
    throw new IllegalStateException("Axis " + name + " is not an enum (was " + v.getClass() + ")");
  }

  private Object require(String name) {
    Object v = params.get(name);
    if (v == null) {
      throw new IllegalStateException("Trial " + trialId + " has no axis named " + name);
    }
    return v;
  }
}
