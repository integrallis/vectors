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

import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BulkRoutesTest {

  @Test
  void reservoirReturnsEveryIndexWhenTakeCoversTotal() {
    assertThat(BulkRoutes.reservoirIndices(4, 4, new Random(1L)))
        .containsExactlyInAnyOrder(0, 1, 2, 3);
  }

  @Test
  void reservoirReturnsRequestedUniqueInBoundsIndices() {
    var picked = BulkRoutes.reservoirIndices(100, 10, new Random(42L));

    assertThat(picked).hasSize(10);
    assertThat(picked).allSatisfy(idx -> assertThat(idx).isBetween(0, 99));
  }

  @Test
  void reservoirUsesProvidedRandomGenerator() {
    var first = BulkRoutes.reservoirIndices(100, 10, new Random(1L));
    var second = BulkRoutes.reservoirIndices(100, 10, new Random(2L));

    assertThat(first).isNotEqualTo(second);
  }
}
