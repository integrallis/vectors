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
package com.integrallis.vectors.studio.core.projection.smile;

import com.integrallis.vectors.studio.core.projection.ProgressListener;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Drives time-based progress estimates from a daemon thread while a blocking projection backend
 * (Smile's {@code TSNE.fit} / {@code UMAP.fit}) is running. Smile does not expose mid-iteration
 * callbacks, so honest "still running" feedback is the best we can do without forking the
 * dependency.
 *
 * <p>The estimate is labeled by passing {@code coords == null} on every heartbeat call to {@link
 * ProgressListener#onIteration}; only the caller's final call (after {@code fit()} returns) carries
 * the real coordinates. UI consumers can distinguish "real coords landed" from "still computing" by
 * inspecting {@code coords}.
 *
 * <p>Iteration count is heartbeat-derived (not algorithm-derived): it increments by one every
 * {@code intervalMillis}, capped at {@code totalIterations - 1} so the terminal call's {@code iter
 * == total} unambiguously marks completion.
 */
public final class SmileProgressHeartbeat implements AutoCloseable {

  private static final long DEFAULT_INTERVAL_MILLIS = 500L;

  private final Thread thread;
  private final AtomicBoolean stop = new AtomicBoolean(false);

  /**
   * Starts a heartbeat thread. The first heartbeat fires after {@code intervalMillis} have elapsed
   * (so a fast {@code fit()} doesn't get a spurious "iter 1" event before its real completion).
   *
   * @param listener the progress listener; null disables the heartbeat
   * @param totalIterations the configured iteration count for the projection
   * @param intervalMillis heartbeat period in millis; {@code <= 0} uses the default
   */
  public static SmileProgressHeartbeat start(
      ProgressListener listener, int totalIterations, long intervalMillis) {
    return new SmileProgressHeartbeat(listener, totalIterations, intervalMillis);
  }

  public static SmileProgressHeartbeat start(ProgressListener listener, int totalIterations) {
    return start(listener, totalIterations, DEFAULT_INTERVAL_MILLIS);
  }

  private SmileProgressHeartbeat(
      ProgressListener listener, int totalIterations, long intervalMillis) {
    long interval = intervalMillis <= 0L ? DEFAULT_INTERVAL_MILLIS : intervalMillis;
    if (listener == null || totalIterations <= 1) {
      // Nothing to report — degenerate case, leave the thread reference unset.
      this.thread = null;
      return;
    }
    int cap = Math.max(1, totalIterations - 1);
    this.thread =
        new Thread(
            () -> {
              int iter = 0;
              while (!stop.get()) {
                try {
                  if (!sleepOrStop(interval)) return;
                  if (stop.get()) return;
                  iter = Math.min(iter + 1, cap);
                  // coords == null signals "interim estimate, not real coordinates"
                  listener.onIteration(iter, totalIterations, null);
                } catch (Throwable ignored) {
                  // A misbehaving listener must not crash the heartbeat — its purpose is best-
                  // effort feedback during a blocking backend call.
                }
              }
            },
            "vectors-smile-progress-heartbeat");
    this.thread.setDaemon(true);
    this.thread.start();
  }

  /** Sleeps for {@code ms} or returns false if stop was requested mid-sleep. */
  private boolean sleepOrStop(long ms) throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(ms);
    while (!stop.get()) {
      long remainingNanos = deadline - System.nanoTime();
      if (remainingNanos <= 0L) return true;
      long sliceMillis = Math.min(50L, TimeUnit.NANOSECONDS.toMillis(remainingNanos) + 1L);
      Thread.sleep(sliceMillis);
    }
    return false;
  }

  @Override
  public void close() {
    stop.set(true);
    if (thread == null) return;
    thread.interrupt();
    try {
      thread.join(200);
    } catch (InterruptedException ie) {
      Thread.currentThread().interrupt();
    }
  }
}
