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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RouteMatchTest {

  @Nested
  class NoMatch {

    @Test
    void nameIsNull() {
      RouteMatch match = RouteMatch.noMatch();
      assertThat(match.getName()).isNull();
    }

    @Test
    void distanceIsNull() {
      RouteMatch match = RouteMatch.noMatch();
      assertThat(match.getDistance()).isNull();
    }
  }

  @Nested
  class Of {

    @Test
    void returnsNameAndDistance() {
      RouteMatch match = RouteMatch.of("politics", 0.15);
      assertThat(match.getName()).isEqualTo("politics");
      assertThat(match.getDistance()).isEqualTo(0.15);
    }

    @Test
    void zeroDistanceIsValid() {
      RouteMatch match = RouteMatch.of("exact", 0.0);
      assertThat(match.getName()).isEqualTo("exact");
      assertThat(match.getDistance()).isEqualTo(0.0);
    }
  }
}
