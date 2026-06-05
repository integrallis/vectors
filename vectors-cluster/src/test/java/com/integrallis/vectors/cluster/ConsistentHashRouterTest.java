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
package com.integrallis.vectors.cluster;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ConsistentHashRouterTest {

  @Test
  void allKeysRouteWithinRange() {
    ConsistentHashRouter router = new ConsistentHashRouter(8);
    for (int i = 0; i < 5_000; i++) {
      int shard = router.route("doc-" + i);
      assertThat(shard).isBetween(0, 7);
    }
  }

  @Test
  void routingIsDeterministicAcrossInstances() {
    ConsistentHashRouter a = new ConsistentHashRouter(5);
    ConsistentHashRouter b = new ConsistentHashRouter(5);
    for (int i = 0; i < 1_000; i++) {
      String id = "key-" + i;
      assertThat(a.route(id)).isEqualTo(b.route(id));
    }
  }

  @Test
  void singleShardRoutesEverythingToZero() {
    ConsistentHashRouter router = new ConsistentHashRouter(1);
    for (int i = 0; i < 1_000; i++) {
      assertThat(router.route("anything-" + i)).isZero();
    }
  }

  @Test
  void distributionIsRoughlyBalanced() {
    int shards = 8;
    int keys = 40_000;
    ConsistentHashRouter router = new ConsistentHashRouter(shards);
    int[] counts = new int[shards];
    for (int i = 0; i < keys; i++) {
      counts[router.route("balanced-" + i)]++;
    }
    double mean = (double) keys / shards;
    for (int s = 0; s < shards; s++) {
      // 200 virtual nodes per shard keeps every shard well within ±35% of the mean.
      assertThat(counts[s])
          .as("shard %d load", s)
          .isBetween((int) (mean * 0.65), (int) (mean * 1.35));
    }
  }

  @Test
  void growingShardCountRemapsOnlyASmallFraction() {
    int keys = 40_000;
    ConsistentHashRouter four = new ConsistentHashRouter(4);
    ConsistentHashRouter five = new ConsistentHashRouter(5);
    int moved = 0;
    for (int i = 0; i < keys; i++) {
      String id = "reshard-" + i;
      if (four.route(id) != five.route(id)) {
        moved++;
      }
    }
    double movedFraction = (double) moved / keys;
    // Consistent hashing moves ~1/5 of keys when going 4 -> 5 shards. Plain hash % N would move
    // roughly 80%; assert we are comfortably under that.
    assertThat(movedFraction).isLessThan(0.35);
  }

  @Test
  void rejectsInvalidArguments() {
    assertThatThrownBy(() -> new ConsistentHashRouter(0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new ConsistentHashRouter(4, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new ConsistentHashRouter(4).route(null))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
