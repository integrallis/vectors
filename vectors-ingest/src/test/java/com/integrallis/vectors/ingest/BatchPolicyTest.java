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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BatchPolicyTest {

  @Test
  void defaultsAreSane() {
    BatchPolicy p = BatchPolicy.defaults();
    assertThat(p.maxDocs()).isEqualTo(1024);
    assertThat(p.maxBytes()).isEqualTo(32L * 1024L * 1024L);
    assertThat(p.maxLatency()).isEqualTo(Duration.ofSeconds(1));
  }

  @Test
  void rejectsNonPositiveDocs() {
    assertThatThrownBy(() -> new BatchPolicy(0, 1024, Duration.ofSeconds(1)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNonPositiveBytes() {
    assertThatThrownBy(() -> new BatchPolicy(1, 0, Duration.ofSeconds(1)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsZeroOrNegativeLatency() {
    assertThatThrownBy(() -> new BatchPolicy(1, 1, Duration.ZERO))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new BatchPolicy(1, 1, Duration.ofSeconds(-1)))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
