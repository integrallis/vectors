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

import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * Concurrency tests for {@link TieredCluster#scan} read-through (P1.8).
 *
 * <p>Pre-fix audit T1.3: the read-through state lived in two independent volatile fields ({@code
 * readThroughBackend} and {@code readThroughCache}) with no synchronization between them, so a cold
 * cluster being probed by N threads issued N concurrent network GETs (one per thread) instead of
 * single-flighting through one fetch. With expensive object-storage backends this is a real cost
 * amplifier; the fix collapses the state into a single atomic snapshot and serializes the
 * cold-load.
 */
class TieredClusterReadThroughRaceTest {

  private static final int DIM = 16;

  @Test
  void concurrentColdProbes_singleFlightTheBackendFetch() throws Exception {
    float[][] vecs = randomVecs(40, DIM, 7L);
    TieredCluster cluster =
        new TieredCluster(makePartition(0, 40), vecs, SimilarityFunction.EUCLIDEAN);
    HeapStorageBackend underlying = new HeapStorageBackend();
    cluster.storeT3(underlying);
    CountingBackend counter = new CountingBackend(underlying);
    cluster.enableReadThrough(counter);

    int threads = 16;
    int probesPerThread = 8;
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(threads);
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      float[] query = randomQuery(DIM, 0xC0FFEEL);
      for (int t = 0; t < threads; t++) {
        pool.submit(
            () -> {
              try {
                start.await();
                for (int p = 0; p < probesPerThread; p++) {
                  List<IvfHit> hits = cluster.scan(query, 5, -Float.MAX_VALUE);
                  assertThat(hits).isNotNull();
                }
              } catch (Exception e) {
                throw new RuntimeException(e);
              } finally {
                done.countDown();
              }
            });
      }
      start.countDown();
      assertThat(done.await(30, TimeUnit.SECONDS)).isTrue();
    } finally {
      pool.shutdownNow();
    }

    // The cluster slice is identical regardless of how many threads fetched it, so the cache must
    // be warm and the backend must have been touched at most once.
    assertThat(cluster.isReadThroughWarm()).isTrue();
    assertThat(counter.gets.get())
        .as("concurrent cold probes must single-flight; observed N=%s fetches", counter.gets.get())
        .isLessThanOrEqualTo(1);
  }

  @Test
  void dropReadThroughCacheDuringScan_doesNotCrashConcurrentScans() throws Exception {
    float[][] vecs = randomVecs(50, DIM, 11L);
    TieredCluster cluster =
        new TieredCluster(makePartition(0, 50), vecs, SimilarityFunction.EUCLIDEAN);
    HeapStorageBackend underlying = new HeapStorageBackend();
    cluster.storeT3(underlying);
    CountingBackend counter = new CountingBackend(underlying);
    cluster.enableReadThrough(counter);

    int scanners = 8;
    int rounds = 200;
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(scanners + 1);
    ExecutorService pool = Executors.newFixedThreadPool(scanners + 1);
    try {
      float[] query = randomQuery(DIM, 0xABCDEL);
      for (int t = 0; t < scanners; t++) {
        pool.submit(
            () -> {
              try {
                start.await();
                for (int r = 0; r < rounds; r++) {
                  assertThat(cluster.scan(query, 5, -Float.MAX_VALUE)).isNotNull();
                }
              } catch (Exception e) {
                throw new RuntimeException(e);
              } finally {
                done.countDown();
              }
            });
      }
      // Dropper races the scanners.
      pool.submit(
          () -> {
            try {
              start.await();
              for (int r = 0; r < rounds; r++) {
                cluster.dropReadThroughCache();
              }
            } catch (Exception e) {
              throw new RuntimeException(e);
            } finally {
              done.countDown();
            }
          });
      start.countDown();
      assertThat(done.await(30, TimeUnit.SECONDS)).isTrue();
    } finally {
      pool.shutdownNow();
    }
    // Bound on backend fetches: at most one per drop round (worst case alternating drop/scan).
    // Without the fix the count could spike to (rounds * scanners) due to multiple racing fetches
    // landing on each cold cycle.
    assertThat(counter.gets.get())
        .as("fetches must be bounded by drop frequency")
        .isLessThanOrEqualTo(rounds + 1);
  }

  // ─── helpers ──────────────────────────────────────────────────────────────

  private static float[][] randomVecs(int n, int dim, long seed) {
    Random rng = new Random(seed);
    float[][] m = new float[n][dim];
    for (float[] row : m) for (int d = 0; d < dim; d++) row[d] = rng.nextFloat() * 2f - 1f;
    return m;
  }

  private static float[] randomQuery(int dim, long seed) {
    Random rng = new Random(seed);
    float[] q = new float[dim];
    for (int d = 0; d < dim; d++) q[d] = rng.nextFloat() * 2f - 1f;
    return q;
  }

  private static ClusterPartition makePartition(int clusterId, int size) {
    int[] ordinals = new int[size];
    for (int i = 0; i < size; i++) ordinals[i] = i;
    return ClusterPartition.of(clusterId, new float[DIM], ordinals);
  }

  /**
   * Storage backend wrapper that counts every {@code get(key)} call. Used to verify single-flight
   * behavior of {@link TieredCluster#scan}'s read-through fetch under concurrency.
   */
  private static final class CountingBackend implements StorageBackend {
    final AtomicInteger gets = new AtomicInteger();
    private final StorageBackend delegate;

    CountingBackend(StorageBackend delegate) {
      this.delegate = delegate;
    }

    @Override
    public void put(String key, byte[] value) throws IOException {
      delegate.put(key, value);
    }

    @Override
    public byte[] get(String key) throws IOException {
      gets.incrementAndGet();
      return delegate.get(key);
    }

    @Override
    public StorageBackend.StoredValue getWithEtag(String key) throws IOException {
      return delegate.getWithEtag(key);
    }

    @Override
    public byte[] getRange(String key, long offset, int length) throws IOException {
      return delegate.getRange(key, offset, length);
    }

    @Override
    public List<String> list(String prefix) throws IOException {
      return delegate.list(prefix);
    }

    @Override
    public void delete(String key) throws IOException {
      delegate.delete(key);
    }

    @Override
    public StorageBackend.ConditionalPutResult conditionalPut(
        String key, byte[] value, String expectedEtag) throws IOException {
      return delegate.conditionalPut(key, value, expectedEtag);
    }
  }
}
