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
package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.ivf.TierPolicy.Tier;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** P13: TierPolicy — frequency-based tier promotion decisions (T0–T3 cascade). */
@Tag("unit")
class TierPolicyTest {

  // ─── desiredTier ─────────────────────────────────────────────────────────

  @Test
  void desiredTier_belowBothThresholds_isT3() {
    TierPolicy policy = new TierPolicy(10, 3);
    assertThat(policy.desiredTier(0)).isEqualTo(Tier.T3);
    assertThat(policy.desiredTier(1)).isEqualTo(Tier.T3);
    assertThat(policy.desiredTier(2)).isEqualTo(Tier.T3);
  }

  @Test
  void desiredTier_atT2Threshold_isT2() {
    TierPolicy policy = new TierPolicy(10, 3);
    assertThat(policy.desiredTier(3)).isEqualTo(Tier.T2);
    assertThat(policy.desiredTier(4)).isEqualTo(Tier.T2);
    assertThat(policy.desiredTier(9)).isEqualTo(Tier.T2);
  }

  @Test
  void desiredTier_atT1Threshold_isT1() {
    TierPolicy policy = new TierPolicy(10, 3);
    assertThat(policy.desiredTier(10)).isEqualTo(Tier.T1);
    assertThat(policy.desiredTier(100)).isEqualTo(Tier.T1);
  }

  @Test
  void desiredTier_monotonic() {
    // Higher access count → same or higher tier (never downgrades)
    TierPolicy policy = new TierPolicy(10, 3);
    int[] ordinal = {0};
    Tier[] tiers = {
      Tier.T3, Tier.T3, Tier.T3, Tier.T2, Tier.T2, Tier.T2,
      Tier.T2, Tier.T2, Tier.T2, Tier.T2, Tier.T1, Tier.T1
    };
    for (int count = 0; count < tiers.length; count++) {
      assertThat(policy.desiredTier(count)).as("count=%d", count).isEqualTo(tiers[count]);
    }
  }

  @Test
  void desiredTier_thresholdsAreInclusive() {
    TierPolicy policy = new TierPolicy(5, 2);
    assertThat(policy.desiredTier(1)).isEqualTo(Tier.T3); // < t2Threshold
    assertThat(policy.desiredTier(2)).isEqualTo(Tier.T2); // == t2Threshold
    assertThat(policy.desiredTier(4)).isEqualTo(Tier.T2); // between
    assertThat(policy.desiredTier(5)).isEqualTo(Tier.T1); // == t1Threshold
  }

  // ─── applyAll ──────────────────────────────────────────────────────────────

  @Test
  void applyAll_mapsEachClusterCorrectly() {
    TierPolicy policy = new TierPolicy(10, 3);
    Map<Integer, Integer> counts = new HashMap<>();
    counts.put(0, 0); // cold
    counts.put(1, 3); // warm
    counts.put(2, 10); // hot
    counts.put(3, 15); // hot

    Map<Integer, Tier> result = policy.applyAll(counts);

    assertThat(result.get(0)).isEqualTo(Tier.T3);
    assertThat(result.get(1)).isEqualTo(Tier.T2);
    assertThat(result.get(2)).isEqualTo(Tier.T1);
    assertThat(result.get(3)).isEqualTo(Tier.T1);
  }

  @Test
  void applyAll_emptyMap_returnsEmpty() {
    TierPolicy policy = TierPolicy.defaults();
    assertThat(policy.applyAll(Map.of())).isEmpty();
  }

  // ─── defaults ─────────────────────────────────────────────────────────────

  @Test
  void defaults_hasReasonableThresholds() {
    TierPolicy policy = TierPolicy.defaults();
    // T3 for cold-start
    assertThat(policy.desiredTier(0)).isEqualTo(Tier.T3);
    // After a few accesses → T2
    assertThat(policy.desiredTier(policy.t2Threshold())).isEqualTo(Tier.T2);
    // After many accesses → T1
    assertThat(policy.desiredTier(policy.t1Threshold())).isEqualTo(Tier.T1);
  }

  @Test
  void defaults_t1ThresholdGreaterThanT2Threshold() {
    TierPolicy policy = TierPolicy.defaults();
    assertThat(policy.t1Threshold()).isGreaterThan(policy.t2Threshold());
  }

  // ─── validation ───────────────────────────────────────────────────────────

  @Test
  void rejectsT1LessThanOrEqualT2Threshold() {
    assertThatThrownBy(() -> new TierPolicy(3, 3)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new TierPolicy(2, 5)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNonPositiveThresholds() {
    assertThatThrownBy(() -> new TierPolicy(10, 0)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new TierPolicy(0, 0)).isInstanceOf(IllegalArgumentException.class);
  }
}
