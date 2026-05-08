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

import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IngestMetricsTest {

  @Test
  void buildsHappyPath() {
    IngestMetrics m = new IngestMetrics(100, 100, 100, 5, 4096L, 2, 16, 0L, Optional.empty());
    assertThat(m.docsRead()).isEqualTo(100);
    assertThat(m.queueCapacity()).isEqualTo(16);
    assertThat(m.lastError()).isEmpty();
  }

  @Test
  void rejectsNegativeCounters() {
    assertThatThrownBy(() -> new IngestMetrics(-1, 0, 0, 0, 0, 0, 0, 0L, Optional.empty()))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void normalisesNullErrorOptional() {
    IngestMetrics m = new IngestMetrics(0, 0, 0, 0, 0, 0, 0, 0L, null);
    assertThat(m.lastError()).isEmpty();
  }
}
