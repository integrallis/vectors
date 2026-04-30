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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CacheAdmissionPolicyTest {

  @Test
  void allowAllAcceptsEverything() {
    CacheAdmissionPolicy<String> policy = CacheAdmissionPolicy.allowAll();
    assertThat(policy.test("hello")).isTrue();
    assertThat(policy.test("")).isTrue();
    assertThat(policy.test(null)).isTrue();
  }

  @Test
  void denyAllRejectsEverything() {
    CacheAdmissionPolicy<String> policy = CacheAdmissionPolicy.denyAll();
    assertThat(policy.test("hello")).isFalse();
    assertThat(policy.test("")).isFalse();
    assertThat(policy.test(null)).isFalse();
  }

  @Test
  void andCombinesPolicies() {
    CacheAdmissionPolicy<String> notNull = v -> v != null;
    CacheAdmissionPolicy<String> notEmpty = v -> v != null && !v.isEmpty();
    CacheAdmissionPolicy<String> combined = notNull.and(notEmpty);

    assertThat(combined.test("hello")).isTrue();
    assertThat(combined.test("")).isFalse();
    assertThat(combined.test(null)).isFalse();
  }

  @Test
  void orCombinesPolicies() {
    CacheAdmissionPolicy<String> startsWithA = v -> v != null && v.startsWith("a");
    CacheAdmissionPolicy<String> startsWithB = v -> v != null && v.startsWith("b");
    CacheAdmissionPolicy<String> combined = startsWithA.or(startsWithB);

    assertThat(combined.test("apple")).isTrue();
    assertThat(combined.test("banana")).isTrue();
    assertThat(combined.test("cherry")).isFalse();
  }

  @Test
  void negateInvertsPolicy() {
    CacheAdmissionPolicy<String> rejectNull = v -> v != null;
    CacheAdmissionPolicy<String> acceptNull = rejectNull.negate();

    assertThat(acceptNull.test(null)).isTrue();
    assertThat(acceptNull.test("hello")).isFalse();
  }

  @Test
  void lambdaWorksAsFunctionalInterface() {
    CacheAdmissionPolicy<Integer> positiveOnly = v -> v != null && v > 0;
    assertThat(positiveOnly.test(42)).isTrue();
    assertThat(positiveOnly.test(-1)).isFalse();
  }
}
