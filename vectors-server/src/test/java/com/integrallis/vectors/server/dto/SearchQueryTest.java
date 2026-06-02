/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Functional Source License, Version 1.1, Apache 2.0 Future License
 * (the "License"); you may not use this file except in compliance with the License.
 *
 *     https://fsl.software/FSL-1.1-ALv2.txt
 *
 * Unless required by applicable law or agreed to in writing, software distributed under
 * the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 * ANY KIND, either express or implied. See the License for the specific language
 * governing permissions and limitations under the License.
 *
 * Change Date: April 25, 2028
 * Change License: Apache License, Version 2.0
 */
package com.integrallis.vectors.server.dto;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SearchQueryTest {

  @Test
  void hybridModeDefaultsToRrf() {
    SearchQuery query =
        new SearchQuery(new float[] {1.0f}, 1, null, null, null, null, null, null, null, null);

    assertThat(query.validate()).isNull();
    assertThat(query.hybridModeDefault()).isEqualTo("RRF");
  }

  @Test
  void weightedHybridModeIsAccepted() {
    SearchQuery query =
        new SearchQuery(
            new float[] {1.0f}, 1, null, null, null, null, null, null, "term", "WEIGHTED");

    assertThat(query.validate()).isNull();
    assertThat(query.hybridModeDefault()).isEqualTo("WEIGHTED");
  }

  @Test
  void unknownHybridModeIsRejected() {
    SearchQuery query =
        new SearchQuery(new float[] {1.0f}, 1, null, null, null, null, null, null, "term", "BOGUS");

    assertThat(query.validate()).isEqualTo("hybridMode must be RRF or WEIGHTED");
  }
}
