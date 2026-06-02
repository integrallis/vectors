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
package com.integrallis.vectors.server.routing;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.hybrid.RRFFusion;
import com.integrallis.vectors.hybrid.WeightedFusion;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SearchRoutesTest {

  @Test
  void rrfHybridModeSelectsRrfFusion() {
    assertThat(SearchRoutes.fusionFor("RRF")).isInstanceOf(RRFFusion.class);
  }

  @Test
  void weightedHybridModeSelectsWeightedFusion() {
    assertThat(SearchRoutes.fusionFor("WEIGHTED")).isInstanceOf(WeightedFusion.class);
  }

  @Test
  void unknownHybridModeIsRejected() {
    assertThatThrownBy(() -> SearchRoutes.fusionFor("BOGUS"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("hybridMode");
  }
}
