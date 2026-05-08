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

import java.time.Duration;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.function.LongConsumer;

/**
 * Bounded exponential-backoff retry with proportional jitter. {@link IllegalArgumentException} and
 * {@link IllegalStateException} bypass retry — they signal programmer or configuration errors that
 * re-running cannot fix.
 *
 * <p>Backoff for attempt {@code i} (0-indexed) before retry: {@code initialBackoff * 2^i}, capped
 * at {@code maxBackoff}, scaled by {@code 1.0 + (rng-0.5) * jitter}.
 */
public final class RetryPolicy {

  private final int maxAttempts;
  private final Duration initialBackoff;
  private final Duration maxBackoff;
  private final double jitter;
  private final Random rng;
  private final LongConsumer sleeper;

  /** Default policy: 3 attempts, 100 ms → 2 s, 20 % jitter, {@link Thread#sleep}. */
  public static RetryPolicy defaults() {
    return new RetryPolicy(
        3, Duration.ofMillis(100), Duration.ofSeconds(2), 0.2, new Random(), null);
  }

  public RetryPolicy(
      int maxAttempts,
      Duration initialBackoff,
      Duration maxBackoff,
      double jitter,
      Random rng,
      LongConsumer sleeper) {
    if (maxAttempts <= 0) {
      throw new IllegalArgumentException("maxAttempts must be > 0");
    }
    Objects.requireNonNull(initialBackoff, "initialBackoff");
    Objects.requireNonNull(maxBackoff, "maxBackoff");
    if (initialBackoff.isNegative() || initialBackoff.isZero()) {
      throw new IllegalArgumentException("initialBackoff must be positive");
    }
    if (maxBackoff.compareTo(initialBackoff) < 0) {
      throw new IllegalArgumentException("maxBackoff must be >= initialBackoff");
    }
    if (jitter < 0.0 || jitter >= 1.0) {
      throw new IllegalArgumentException("jitter must be in [0, 1)");
    }
    this.maxAttempts = maxAttempts;
    this.initialBackoff = initialBackoff;
    this.maxBackoff = maxBackoff;
    this.jitter = jitter;
    this.rng = Objects.requireNonNull(rng, "rng");
    this.sleeper = sleeper != null ? sleeper : RetryPolicy::defaultSleep;
  }

  /** Convenience constructor that uses {@link Thread#sleep} and a fresh {@link Random}. */
  public RetryPolicy(int maxAttempts, Duration initialBackoff, Duration maxBackoff, double jitter) {
    this(maxAttempts, initialBackoff, maxBackoff, jitter, new Random(), null);
  }

  public int maxAttempts() {
    return maxAttempts;
  }

  /**
   * Returns the (deterministic when {@code rng} is seeded) backoff for the {@code attempt}-th
   * retry.
   */
  public Duration backoffFor(int attempt) {
    if (attempt < 0) {
      throw new IllegalArgumentException("attempt must be >= 0");
    }
    long base = initialBackoff.toMillis();
    long capped = maxBackoff.toMillis();
    long shifted = (attempt >= 62) ? capped : Math.min(capped, base << attempt);
    double scale = 1.0 + (rng.nextDouble() - 0.5) * jitter;
    long jittered = Math.max(0L, Math.round(shifted * scale));
    return Duration.ofMillis(jittered);
  }

  /**
   * Executes {@code task} up to {@link #maxAttempts} times, sleeping {@link #backoffFor(int)}
   * between attempts. Re-throws the last exception (wrapping checked exceptions in {@link
   * RuntimeException}) when every attempt fails.
   */
  public <T> T execute(Callable<T> task) throws Exception {
    Exception last = null;
    for (int attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        return task.call();
      } catch (IllegalArgumentException | IllegalStateException nonRetryable) {
        throw nonRetryable;
      } catch (Exception e) {
        last = e;
        if (attempt + 1 == maxAttempts) {
          break;
        }
        sleeper.accept(backoffFor(attempt).toMillis());
      }
    }
    throw last;
  }

  private static void defaultSleep(long millis) {
    if (millis <= 0) return;
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("retry sleep interrupted", e);
    }
  }
}
