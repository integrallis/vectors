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

import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RetryPolicyTest {

  @Test
  void backoffGrowsExponentiallyAndCaps() {
    // Deterministic RNG (always 0.5 → scale 1.0 with default jitter), no real sleeping.
    RetryPolicy p =
        new RetryPolicy(
            5, Duration.ofMillis(100), Duration.ofSeconds(2), 0.0, new Random(0L), ms -> {});
    assertThat(p.backoffFor(0)).isEqualTo(Duration.ofMillis(100));
    assertThat(p.backoffFor(1)).isEqualTo(Duration.ofMillis(200));
    assertThat(p.backoffFor(2)).isEqualTo(Duration.ofMillis(400));
    assertThat(p.backoffFor(3)).isEqualTo(Duration.ofMillis(800));
    // 100 << 4 = 1600 < 2000 cap
    assertThat(p.backoffFor(4)).isEqualTo(Duration.ofMillis(1600));
    // 100 << 5 = 3200 → capped at 2000
    assertThat(p.backoffFor(5)).isEqualTo(Duration.ofMillis(2000));
    // very large attempts must not overflow
    assertThat(p.backoffFor(60)).isEqualTo(Duration.ofMillis(2000));
    assertThat(p.backoffFor(80)).isEqualTo(Duration.ofMillis(2000));
  }

  @Test
  void jitterStaysWithinBounds() {
    RetryPolicy p =
        new RetryPolicy(
            1, Duration.ofMillis(1000), Duration.ofMillis(1000), 0.4, new Random(42L), ms -> {});
    for (int i = 0; i < 200; i++) {
      long ms = p.backoffFor(0).toMillis();
      // jitter=0.4 → scale ∈ [0.8, 1.2]
      assertThat(ms).isBetween(800L, 1200L);
    }
  }

  @Test
  void retriesUpToMaxAttemptsThenThrowsLast() {
    AtomicInteger calls = new AtomicInteger();
    List<Long> sleeps = new ArrayList<>();
    RetryPolicy p =
        new RetryPolicy(
            3, Duration.ofMillis(10), Duration.ofMillis(40), 0.0, new Random(0L), sleeps::add);
    assertThatThrownBy(
            () ->
                p.execute(
                    () -> {
                      calls.incrementAndGet();
                      throw new IOException("transient #" + calls.get());
                    }))
        .isInstanceOf(IOException.class)
        .hasMessage("transient #3");
    assertThat(calls.get()).isEqualTo(3);
    // sleeps between attempts: 10, 20 (cap not yet hit)
    assertThat(sleeps).containsExactly(10L, 20L);
  }

  @Test
  void successOnSecondAttempt() throws Exception {
    AtomicInteger calls = new AtomicInteger();
    RetryPolicy p =
        new RetryPolicy(
            3, Duration.ofMillis(10), Duration.ofMillis(20), 0.0, new Random(0L), ms -> {});
    String r =
        p.execute(
            () -> {
              if (calls.incrementAndGet() < 2) throw new IOException("transient");
              return "ok";
            });
    assertThat(r).isEqualTo("ok");
    assertThat(calls.get()).isEqualTo(2);
  }

  @Test
  void illegalArgumentBypassesRetry() {
    AtomicInteger calls = new AtomicInteger();
    RetryPolicy p =
        new RetryPolicy(
            5, Duration.ofMillis(1), Duration.ofMillis(1), 0.0, new Random(0L), ms -> {});
    assertThatThrownBy(
            () ->
                p.execute(
                    () -> {
                      calls.incrementAndGet();
                      throw new IllegalArgumentException("bad");
                    }))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(calls.get()).isEqualTo(1);
  }

  @Test
  void illegalStateBypassesRetry() {
    AtomicInteger calls = new AtomicInteger();
    RetryPolicy p =
        new RetryPolicy(
            5, Duration.ofMillis(1), Duration.ofMillis(1), 0.0, new Random(0L), ms -> {});
    assertThatThrownBy(
            () ->
                p.execute(
                    () -> {
                      calls.incrementAndGet();
                      throw new IllegalStateException("bad");
                    }))
        .isInstanceOf(IllegalStateException.class);
    assertThat(calls.get()).isEqualTo(1);
  }

  @Test
  void rejectsBadConstructorArgs() {
    Random rng = new Random(0L);
    assertThatThrownBy(
            () ->
                new RetryPolicy(0, Duration.ofMillis(1), Duration.ofMillis(1), 0.0, rng, ms -> {}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                new RetryPolicy(1, Duration.ofMillis(10), Duration.ofMillis(5), 0.0, rng, ms -> {}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                new RetryPolicy(1, Duration.ofMillis(1), Duration.ofMillis(1), 1.0, rng, ms -> {}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                new RetryPolicy(1, Duration.ofMillis(1), Duration.ofMillis(1), -0.1, rng, ms -> {}))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
