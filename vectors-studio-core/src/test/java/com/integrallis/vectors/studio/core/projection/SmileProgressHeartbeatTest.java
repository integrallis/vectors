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
package com.integrallis.vectors.studio.core.projection;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.studio.core.projection.smile.SmileProgressHeartbeat;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Pins the heartbeat contract used by the Smile T-SNE/UMAP wrappers (audit T4.12):
 *
 * <ul>
 *   <li>While the heartbeat is running, every emitted {@link ProgressListener#onIteration} call
 *       carries {@code coords == null} (labeled as a time-based estimate, not a real iteration).
 *   <li>{@code iter} is monotonically non-decreasing and capped at {@code totalIterations - 1}.
 *   <li>The heartbeat is a daemon (no thread leaks) and stops promptly on {@link
 *       SmileProgressHeartbeat#close()}.
 * </ul>
 */
@SuppressWarnings("try") // try-with-resources controls heartbeat lifetime; body intentionally
// doesn't reference it
@Tag("unit")
class SmileProgressHeartbeatTest {

  @Test
  void heartbeatEmitsEstimatesWithNullCoordsAndStopsOnClose() throws Exception {
    List<int[]> events = new ArrayList<>();
    List<float[][]> coordsSeen = new ArrayList<>();
    ProgressListener listener =
        new ProgressListener() {
          @Override
          public void onIteration(int iter, int total, float[][] coords) {
            synchronized (events) {
              events.add(new int[] {iter, total});
              coordsSeen.add(coords);
            }
          }
        };

    int total = 200;
    long interval = 30L;
    try (var hb =
        com.integrallis.vectors.studio.core.projection.smile.SmileProgressHeartbeat.start(
            listener, total, interval)) {
      // Sleep ≈ 6 intervals so we observe several heartbeats.
      Thread.sleep(interval * 6L);
    }
    // After close, no further events may arrive — capture state, sleep, re-check.
    int countAfterClose;
    synchronized (events) {
      countAfterClose = events.size();
    }
    Thread.sleep(interval * 4L);
    synchronized (events) {
      assertThat(events.size())
          .as("close() must stop further heartbeats")
          .isEqualTo(countAfterClose);
    }

    synchronized (events) {
      assertThat(events).as("at least a few heartbeats must have fired").hasSizeGreaterThan(2);
      int prev = 0;
      for (int i = 0; i < events.size(); i++) {
        int iter = events.get(i)[0];
        int seenTotal = events.get(i)[1];
        assertThat(seenTotal).isEqualTo(total);
        assertThat(iter).as("iter must be > 0").isGreaterThan(0);
        assertThat(iter).as("iter must stay below total during estimation").isLessThan(total);
        assertThat(iter)
            .as("iter must be monotonically non-decreasing")
            .isGreaterThanOrEqualTo(prev);
        prev = iter;
        assertThat(coordsSeen.get(i)).as("heartbeat estimates must carry coords == null").isNull();
      }
    }
  }

  @Test
  void heartbeatIsNoopForNullListener() throws Exception {
    try (var hb =
        com.integrallis.vectors.studio.core.projection.smile.SmileProgressHeartbeat.start(
            null, 100, 50L)) {
      Thread.sleep(120);
    }
    // No exceptions, no thread leaks — the close call is the assertion.
  }

  @Test
  void heartbeatIsNoopForDegenerateIterationCount() throws Exception {
    List<int[]> events = new ArrayList<>();
    ProgressListener listener = (i, t, c) -> events.add(new int[] {i, t});
    try (var hb =
        com.integrallis.vectors.studio.core.projection.smile.SmileProgressHeartbeat.start(
            listener, 1, 30L)) {
      Thread.sleep(120);
    }
    assertThat(events).as("totalIterations <= 1 leaves no room for heartbeat ticks").isEmpty();
  }
}
