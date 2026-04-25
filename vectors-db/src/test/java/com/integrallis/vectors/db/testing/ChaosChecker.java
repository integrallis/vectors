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

/**
 * Continuous health checker for chaos engineering tests.
 *
 * <p>Inspired by Milvus's checker module. Each checker runs on its own thread (typically a virtual
 * thread), repeatedly executing an operation against the collection and tracking success/failure
 * counts. After the chaos fault is cleared, {@link #assertHealthy()} verifies that all operations
 * resumed correctly.
 *
 * <p>Usage pattern (3-phase Milvus style):
 *
 * <pre>{@code
 * SearchChecker checker = new SearchChecker(collection, query, k);
 * Thread thread = Thread.ofVirtual().start(checker);
 *
 * // Phase 1: steady state — verify operations succeed
 * Thread.sleep(1000);
 * assertThat(checker.successCount()).isGreaterThan(0);
 *
 * // Phase 2: inject fault
 * injectFault();
 * Thread.sleep(2000);
 *
 * // Phase 3: clear fault, verify recovery
 * clearFault();
 * Thread.sleep(1000);
 * checker.stop();
 * thread.join(5000);
 * checker.assertHealthy();
 * }</pre>
 */
public interface ChaosChecker extends Runnable {

  /** Human-readable name (e.g., "search", "insert"). */
  String name();

  /** Number of successful operations since start. */
  int successCount();

  /** Number of failed operations since start. */
  int failureCount();

  /** Signals the checker to stop its run loop. */
  void stop();

  /**
   * Asserts that the checker is in a healthy state after fault recovery. Implementations should
   * verify that recent operations succeeded and that no unrecoverable errors occurred.
   *
   * @throws AssertionError if health checks fail
   */
  void assertHealthy();
}
