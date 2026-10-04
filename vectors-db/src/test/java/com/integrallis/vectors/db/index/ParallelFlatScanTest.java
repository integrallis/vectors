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
package com.integrallis.vectors.db.index;

import static org.junit.jupiter.api.Assertions.*;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ParallelFlatScanTest {
  @Test
  void queryBudgetRemembersRecentContentionAndRecoversWithoutSleeping() {
    var now = new java.util.concurrent.atomic.AtomicLong(1_000_000_000L);
    var budget = new FlatScanAdapter.QueryBudget(now::get);
    assertTrue(budget.enter());
    assertFalse(budget.enter());
    assertTrue(budget.contended());
    now.addAndGet(100_000_000L);
    budget.exit();
    budget.exit();
    // A new burst arriving immediately after the old one must not expand its first query.
    assertFalse(budget.enter());
    budget.exit();
    now.addAndGet(49_000_000L);
    assertFalse(budget.enter());
    budget.exit();
    now.addAndGet(1_000_000L);
    assertTrue(budget.enter());
    assertFalse(budget.contended());
    budget.exit();
    assertTrue(budget.enter());
    budget.exit();
  }

  @Test
  void collectionBatchContextKeepsSerialScansAndRestoresBudgetAfterFailure() {
    int n = 40001;
    float[] row = new float[128];
    Arrays.fill(row, .1f);
    float[][] rows = new float[n][];
    Arrays.fill(rows, row);
    var index = new FlatScanAdapter();
    index.build(rows, SimilarityFunction.COSINE);
    var bean =
        (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory.getThreadMXBean();
    org.junit.jupiter.api.Assumptions.assumeTrue(bean.isThreadAllocatedMemorySupported());
    bean.setThreadAllocatedMemoryEnabled(true);
    FlatScanAdapter.withConcurrentQueries(
        () -> {
          for (int warm = 0; warm < 10; warm++) index.search(row, 10, 100, 1f);
          long tid = Thread.currentThread().threadId(), before = bean.getThreadAllocatedBytes(tid);
          index.search(row, 10, 100, 1f);
          assertTrue(bean.getThreadAllocatedBytes(tid) - before < 2L * n);
          return null;
        });
    assertThrows(
        IllegalStateException.class,
        () ->
            FlatScanAdapter.withConcurrentQueries(
                () -> {
                  throw new IllegalStateException("fixture");
                }));
    assertEquals(0, FlatScanAdapter.CONCURRENT_BATCHES.get(), "batch context leaked after failure");
  }

  @Test
  void competingQueriesDoNotAllocateAnotherFullScoreBuffer() {
    int n = 40001;
    float[] row = new float[128];
    Arrays.fill(row, .1f);
    float[][] rows = new float[n][];
    Arrays.fill(rows, row);
    var index = new FlatScanAdapter();
    index.build(rows, SimilarityFunction.COSINE);
    var bean =
        (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory.getThreadMXBean();
    org.junit.jupiter.api.Assumptions.assumeTrue(bean.isThreadAllocatedMemorySupported());
    bean.setThreadAllocatedMemoryEnabled(true);
    assertTrue(FlatScanAdapter.PARALLEL_SCANS.tryAcquire());
    try {
      for (int warm = 0; warm < 10; warm++) index.search(row, 10, 100, 1f);
      long id = Thread.currentThread().threadId();
      long before = bean.getThreadAllocatedBytes(id);
      index.search(row, 10, 100, 1f);
      long allocated = bean.getThreadAllocatedBytes(id) - before;
      System.out.println("CONTENDED_SCAN_ALLOCATION=" + allocated);
      assertTrue(allocated < 2L * n, "competing scan allocated " + allocated + " bytes");
    } finally {
      FlatScanAdapter.PARALLEL_SCANS.release();
    }
  }

  @Test
  void parallelScoresPreserveOrdinalOrderMetricsAndMutations() {
    var random = new SplittableRandom(635);
    float[][] rows = new float[4099][129];
    float[] query = new float[129];
    for (var row : rows)
      for (int d = 0; d < row.length; d++) row[d] = (float) random.nextDouble(-.08, .08);
    for (int pass = 0; pass < 3; pass++) {
      for (int d = 0; d < query.length; d++) query[d] = (float) random.nextDouble(-.08, .08);
      for (var metric : SimilarityFunction.values()) {
        float[] scores = FlatScanAdapter.scoreAll(query, rows, metric);
        for (int i = 0; i < rows.length; i++)
          assertEquals(metric.compare(query, rows[i]), scores[i], 1e-6f);
      }
      Arrays.fill(rows[137], .031f);
    }
  }

  @Test
  void largeSearchPreservesTieOrderAndConcurrentQueryIsolation() throws Exception {
    int n = 40001, dim = 128;
    float[][] rows = new float[n][];
    float[] best = new float[dim], worst = new float[dim];
    Arrays.fill(best, .1f);
    Arrays.fill(worst, -.1f);
    for (int i = 0; i < n; i++) rows[i] = (i % 3 == 0) ? best : worst;
    var large = new FlatScanAdapter();
    large.build(rows, SimilarityFunction.COSINE);
    var small = new FlatScanAdapter();
    small.build(Arrays.copyOf(rows, 40), SimilarityFunction.COSINE);
    // All admitted top-k entries occur in the first 40 rows. Subsequent equal scores must never
    // change the bounded heap's tie order; this compares the automatic route to a serial scan.
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var jobs = new java.util.ArrayList<java.util.concurrent.Future<?>>();
      for (int task = 0; task < 16; task++) {
        final float[] q = task % 2 == 0 ? best : worst;
        jobs.add(
            executor.submit(
                () -> {
                  var expected = small.search(q, 10, 100, 1f);
                  var actual = large.search(q, 10, 100, 1f);
                  assertArrayEquals(expected.ordinals(), actual.ordinals());
                  assertArrayEquals(expected.scores(), actual.scores());
                }));
      }
      for (var job : jobs) job.get();
    }
    Arrays.fill(best, .2f);
    assertArrayEquals(
        small.search(worst, 10, 100, 1f).ordinals(), large.search(worst, 10, 100, 1f).ordinals());
  }
}
