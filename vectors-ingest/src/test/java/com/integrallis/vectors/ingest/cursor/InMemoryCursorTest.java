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
package com.integrallis.vectors.ingest.cursor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class InMemoryCursorTest {

  @Test
  void unknownSourceReturnsZero() {
    InMemoryCursor c = new InMemoryCursor();
    assertThat(c.load("missing")).isEqualTo(0L);
  }

  @Test
  void roundTrips() throws Exception {
    InMemoryCursor c = new InMemoryCursor();
    c.save("a", 100L);
    c.save("b", 250L);
    assertThat(c.load("a")).isEqualTo(100L);
    assertThat(c.load("b")).isEqualTo(250L);
  }

  @Test
  void overwritesPreviousOffset() throws Exception {
    InMemoryCursor c = new InMemoryCursor();
    c.save("a", 10L);
    c.save("a", 25L);
    assertThat(c.load("a")).isEqualTo(25L);
  }

  @Test
  void rejectsNegativeOffset() {
    InMemoryCursor c = new InMemoryCursor();
    assertThatThrownBy(() -> c.save("a", -1L)).isInstanceOf(IllegalArgumentException.class);
  }
}
