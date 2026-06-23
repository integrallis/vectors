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
package com.integrallis.vectors.db;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * Pins the bounded-wait contract on the per-collection writer lock: when a commit-path component
 * (here a {@link GenerationSubscriber} fired under the lock) blocks, concurrent writers must time
 * out with a {@link WriterLockTimeoutException} instead of parking on a bare {@code
 * ReentrantLock.lock()} forever.
 *
 * <p>Regression coverage for the audit finding "writerLock is a bare ReentrantLock with no timeout
 * — a stalled fsync blocks all commits indefinitely; no escape hatch."
 */
class WriterLockTimeoutTest {

  @Test
  void writerLock_timesOut_whenAnotherWriterBlocksInCommitPath() throws Exception {
    CountDownLatch insideSubscriber = new CountDownLatch(1);
    CountDownLatch releaseSubscriber = new CountDownLatch(1);
    GenerationSubscriber blocker =
        event -> {
          insideSubscriber.countDown();
          try {
            releaseSubscriber.await(30, TimeUnit.SECONDS);
          } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
          }
        };

    // GenerationSubscribers are fired only on persistent commits — so the test runs against a
    // persistent collection rooted in a tempdir.
    Path storageRoot = Files.createTempDirectory("writer-lock-timeout-it-");
    VectorCollection c =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .storagePath(storageRoot.toAbsolutePath())
            .replicateTo(blocker)
            .build();
    try {
      // Shorten the timeout so the test does not have to wait the production default to assert.
      ((VectorCollectionImpl) c).setWriterLockTimeoutMillisForTesting(200L);
      c.add(Document.of("first", new float[] {1, 0, 0, 0}));
      // Drive a commit on another thread — its subscriber blocks while it holds the writer lock.
      AtomicReference<Throwable> committerFailure = new AtomicReference<>();
      Thread committer =
          Thread.startVirtualThread(
              () -> {
                try {
                  c.commit();
                } catch (Throwable t) {
                  committerFailure.set(t);
                }
              });
      assertThat(insideSubscriber.await(5, TimeUnit.SECONDS))
          .as("the committer must reach the subscriber callback")
          .isTrue();

      long t0 = System.nanoTime();
      assertThatThrownBy(() -> c.add(Document.of("second", new float[] {0, 1, 0, 0})))
          .as("a concurrent writer must time out instead of blocking on the held lock")
          .isInstanceOf(WriterLockTimeoutException.class);
      long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
      assertThat(elapsedMs)
          .as("must time out close to the configured 200ms, not park forever")
          .isLessThan(2_000L);

      // Now release the held lock and let the committer finish.
      releaseSubscriber.countDown();
      committer.join(5_000);
      assertThat(committer.isAlive()).isFalse();
      assertThat(committerFailure.get())
          .as("committer should commit successfully once subscriber unblocks")
          .isNull();
    } finally {
      releaseSubscriber.countDown();
      c.close();
      // Best-effort tempdir cleanup.
      try (java.util.stream.Stream<Path> walk = Files.walk(storageRoot)) {
        walk.sorted(Comparator.reverseOrder())
            .forEach(
                p -> {
                  try {
                    Files.deleteIfExists(p);
                  } catch (Exception ignored) {
                    // best-effort
                  }
                });
      } catch (Exception ignored) {
        // best-effort
      }
    }
  }
}
