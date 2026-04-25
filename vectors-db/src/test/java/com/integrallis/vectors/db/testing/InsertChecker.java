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
package com.integrallis.vectors.db.testing;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.db.VectorCollection;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * {@link ChaosChecker} that continuously inserts documents into a collection, tracking success and
 * failure rates.
 */
public final class InsertChecker implements ChaosChecker {

  private final VectorCollection collection;
  private final int dimension;
  private final Random rng;
  private final AtomicInteger successes = new AtomicInteger();
  private final AtomicInteger failures = new AtomicInteger();
  private final AtomicBoolean stopped = new AtomicBoolean(false);
  private final AtomicInteger idCounter = new AtomicInteger();

  public InsertChecker(VectorCollection collection, int dimension, long seed) {
    this.collection = collection;
    this.dimension = dimension;
    this.rng = new Random(seed);
  }

  @Override
  public String name() {
    return "insert";
  }

  @Override
  public int successCount() {
    return successes.get();
  }

  @Override
  public int failureCount() {
    return failures.get();
  }

  @Override
  public void stop() {
    stopped.set(true);
  }

  @Override
  public void run() {
    while (!stopped.get()) {
      try {
        String id = "chaos-" + idCounter.getAndIncrement();
        float[] vector = VectorSearchTestSupport.randomVector(dimension, rng);
        collection.add(new Document(id, vector, null, null));
        successes.incrementAndGet();
      } catch (
          @SuppressWarnings("PMD.AvoidCatchingGenericException")
          Exception e) {
        failures.incrementAndGet();
      }
    }
  }

  @Override
  public void assertHealthy() {
    assertThat(successCount())
        .as("InsertChecker must have completed at least one successful insert")
        .isGreaterThan(0);
  }
}
