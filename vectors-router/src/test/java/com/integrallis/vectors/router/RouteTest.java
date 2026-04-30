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
package com.integrallis.vectors.router;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RouteTest {

  @Nested
  class Builder {

    @Test
    void buildsWithAllFields() {
      Route route =
          Route.builder()
              .name("off-topic")
              .references(List.of("hello", "world"))
              .distanceThreshold(0.5)
              .build();

      assertThat(route.getName()).isEqualTo("off-topic");
      assertThat(route.getReferences()).containsExactly("hello", "world");
      assertThat(route.getDistanceThreshold()).isEqualTo(0.5);
    }

    @Test
    void defaultThresholdIs03() {
      Route route = Route.builder().name("test").references(List.of("a")).build();

      assertThat(route.getDistanceThreshold()).isEqualTo(0.3);
    }

    @Test
    void defaultReferencesIsEmpty() {
      Route route = Route.builder().name("test").build();

      assertThat(route.getReferences()).isEmpty();
    }

    @Test
    void nullNameThrows() {
      assertThatThrownBy(() -> Route.builder().references(List.of("a")).build())
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("name");
    }

    @Test
    void nullReferencesThrows() {
      assertThatThrownBy(() -> Route.builder().name("test").references(null).build())
          .isInstanceOf(NullPointerException.class);
    }
  }

  @Nested
  class Immutability {

    @Test
    void referencesListIsUnmodifiable() {
      Route route = Route.builder().name("test").references(List.of("a", "b")).build();

      assertThatThrownBy(() -> route.getReferences().add("c"))
          .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void mutatingSourceListDoesNotAffectRoute() {
      var refs = new java.util.ArrayList<>(List.of("a", "b"));
      Route route = Route.builder().name("test").references(refs).build();

      refs.add("c");

      assertThat(route.getReferences()).containsExactly("a", "b");
    }
  }
}
