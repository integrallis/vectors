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
package com.integrallis.vectors.ingest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IngestResultTest {

  @Test
  void buildsHappyPath() {
    IngestResult r =
        new IngestResult(10, 10, 10, 1, 1024L, Duration.ofSeconds(2), 10L, Optional.empty());
    assertThat(r.docsCommitted()).isEqualTo(10L);
    assertThat(r.totalDuration()).isEqualTo(Duration.ofSeconds(2));
    assertThat(r.firstError()).isEmpty();
  }

  @Test
  void rejectsNegativeCounters() {
    assertThatThrownBy(() -> new IngestResult(-1, 0, 0, 0, 0, Duration.ZERO, 0L, Optional.empty()))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNegativeDuration() {
    assertThatThrownBy(
            () -> new IngestResult(0, 0, 0, 0, 0, Duration.ofSeconds(-1), 0L, Optional.empty()))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void normalisesNullErrorOptional() {
    IngestResult r = new IngestResult(0, 0, 0, 0, 0, Duration.ZERO, 0L, null);
    assertThat(r.firstError()).isEmpty();
  }
}
