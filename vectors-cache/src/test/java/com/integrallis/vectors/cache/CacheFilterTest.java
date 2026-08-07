/*
 * Copyright 2026 Integrallis Software, LLC
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

import java.util.Map;
import org.junit.jupiter.api.Test;

class CacheFilterTest {

  @Test
  void anyAcceptsEverything() {
    assertThat(CacheFilter.any().test(Map.of())).isTrue();
    assertThat(CacheFilter.any().test(Map.of("model", "gpt-5"))).isTrue();
  }

  @Test
  void matchingRequiresAnExactAttributeValue() {
    CacheFilter filter = CacheFilter.matching("model", "gpt-5");

    assertThat(filter.test(Map.of("model", "gpt-5"))).isTrue();
    assertThat(filter.test(Map.of("model", "claude-opus-5"))).isFalse();
    assertThat(filter.test(Map.of())).isFalse();
  }

  @Test
  void allRequiresEveryAttribute() {
    CacheFilter filter = CacheFilter.all(Map.of("model", "gpt-5", "temperature", "0.0"));

    assertThat(filter.test(Map.of("model", "gpt-5", "temperature", "0.0"))).isTrue();
    assertThat(filter.test(Map.of("model", "gpt-5", "temperature", "0.9"))).isFalse();
    assertThat(filter.test(Map.of("model", "gpt-5"))).isFalse();
  }

  @Test
  void ignoresAttributesItDoesNotConstrain() {
    // A filter names what must match; anything else on the entry is free to vary.
    CacheFilter filter = CacheFilter.matching("tenant", "acme");

    assertThat(filter.test(Map.of("tenant", "acme", "model", "anything"))).isTrue();
  }

  @Test
  void combinesWithAnd() {
    CacheFilter filter =
        CacheFilter.matching("tenant", "acme").and(CacheFilter.matching("model", "gpt-5"));

    assertThat(filter.test(Map.of("tenant", "acme", "model", "gpt-5"))).isTrue();
    assertThat(filter.test(Map.of("tenant", "acme", "model", "other"))).isFalse();
  }
}
