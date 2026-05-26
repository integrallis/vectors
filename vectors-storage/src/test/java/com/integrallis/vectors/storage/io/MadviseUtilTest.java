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
package com.integrallis.vectors.storage.io;

import static org.assertj.core.api.Assertions.assertThatCode;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class MadviseUtilTest {

  @Test
  void isAvailableDoesNotThrow() {
    assertThatCode(MadviseUtil::isAvailable).doesNotThrowAnyException();
  }

  @Test
  void noneStrategyReturnsBeforeTouchingSegment() {
    assertThatCode(() -> MadviseUtil.apply(null, MadviseStrategy.NONE)).doesNotThrowAnyException();
  }

  @Test
  void applyDoesNotSurfaceNativeFailures() {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment segment = arena.allocate(4096);

      for (MadviseStrategy strategy : MadviseStrategy.values()) {
        assertThatCode(() -> MadviseUtil.apply(segment, strategy)).doesNotThrowAnyException();
      }
    }
  }
}
