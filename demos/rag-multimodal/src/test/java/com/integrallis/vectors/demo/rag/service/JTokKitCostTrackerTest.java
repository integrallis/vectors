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
package com.integrallis.vectors.demo.rag.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JTokKitCostTrackerTest {

  @Nested
  @DisplayName("formatCost")
  class FormatCostTest {

    @Test
    @DisplayName("zero cost displays as $0.00")
    void zeroCostDisplaysCorrectly() {
      assertThat(JTokKitCostTracker.formatCost(0.0)).isEqualTo("$0.00");
    }

    @Test
    @DisplayName("sub-cent cost shows 4 decimal places")
    void subCentCostShowsFourDecimals() {
      assertThat(JTokKitCostTracker.formatCost(0.0001)).isEqualTo("$0.0001");
      assertThat(JTokKitCostTracker.formatCost(0.0050)).isEqualTo("$0.0050");
    }

    @Test
    @DisplayName("normal cost shows 2 decimal places")
    void normalCostShowsTwoDecimals() {
      assertThat(JTokKitCostTracker.formatCost(1.50)).isEqualTo("$1.50");
      assertThat(JTokKitCostTracker.formatCost(0.05)).isEqualTo("$0.05");
    }
  }

  @Nested
  @DisplayName("formatTokens")
  class FormatTokensTest {

    @Test
    @DisplayName("small token count shows exact number")
    void smallTokenCount() {
      assertThat(JTokKitCostTracker.formatTokens(42)).isEqualTo("42 tokens");
      assertThat(JTokKitCostTracker.formatTokens(999)).isEqualTo("999 tokens");
    }

    @Test
    @DisplayName("thousands show K suffix")
    void thousandsShowK() {
      assertThat(JTokKitCostTracker.formatTokens(1500)).isEqualTo("1.5K tokens");
      assertThat(JTokKitCostTracker.formatTokens(10000)).isEqualTo("10.0K tokens");
    }

    @Test
    @DisplayName("millions show M suffix")
    void millionsShowM() {
      assertThat(JTokKitCostTracker.formatTokens(1_500_000)).isEqualTo("1.50M tokens");
    }
  }
}
