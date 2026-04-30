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
package com.integrallis.vectors.cache;

import java.util.Objects;

/**
 * Determines whether a value should be admitted to a cache. Used by {@link SemanticCache}
 * implementations to reject values that should not be stored (e.g. LLM error/refusal responses).
 *
 * <p>This is a {@link FunctionalInterface} so callers can supply a lambda. Composition methods
 * ({@link #and}, {@link #or}, {@link #negate}) follow the same pattern as {@link
 * java.util.function.Predicate}.
 *
 * @param <V> the value type
 */
@FunctionalInterface
public interface CacheAdmissionPolicy<V> {

  /**
   * Returns {@code true} if the value should be admitted to the cache, {@code false} if it should
   * be rejected.
   */
  boolean test(V value);

  /**
   * Returns a composed policy that admits a value only if <em>both</em> this policy and {@code
   * other} admit it.
   */
  default CacheAdmissionPolicy<V> and(CacheAdmissionPolicy<? super V> other) {
    Objects.requireNonNull(other, "other");
    return v -> test(v) && other.test(v);
  }

  /**
   * Returns a composed policy that admits a value if <em>either</em> this policy or {@code other}
   * admits it.
   */
  default CacheAdmissionPolicy<V> or(CacheAdmissionPolicy<? super V> other) {
    Objects.requireNonNull(other, "other");
    return v -> test(v) || other.test(v);
  }

  /** Returns a policy that inverts this policy's decision. */
  default CacheAdmissionPolicy<V> negate() {
    return v -> !test(v);
  }

  /** Returns a policy that admits every value. */
  @SuppressWarnings("unchecked")
  static <V> CacheAdmissionPolicy<V> allowAll() {
    return (CacheAdmissionPolicy<V>) AllowAll.INSTANCE;
  }

  /** Returns a policy that rejects every value. */
  @SuppressWarnings("unchecked")
  static <V> CacheAdmissionPolicy<V> denyAll() {
    return (CacheAdmissionPolicy<V>) DenyAll.INSTANCE;
  }
}

/** Singleton allow-all policy. */
enum AllowAll implements CacheAdmissionPolicy<Object> {
  INSTANCE;

  @Override
  public boolean test(Object value) {
    return true;
  }
}

/** Singleton deny-all policy. */
enum DenyAll implements CacheAdmissionPolicy<Object> {
  INSTANCE;

  @Override
  public boolean test(Object value) {
    return false;
  }
}
