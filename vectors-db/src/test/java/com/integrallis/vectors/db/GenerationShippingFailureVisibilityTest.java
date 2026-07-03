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

import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.storage.backend.StorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend.ConditionalPutResult;
import com.integrallis.vectors.storage.backend.StorageBackend.StoredValue;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Pins the observability contract added to {@link GenerationShippingSubscriber}: a put-failure on
 * the destination backend must increment {@code shipFailureCount}, expose {@code lastShipFailure},
 * and invoke an optional failure callback with the affected generation number. Replicas can then
 * surface "follower fell behind" through metrics instead of relying on log scraping.
 */
class GenerationShippingFailureVisibilityTest {

  @Test
  void putFailureBumpsCounterAndFiresCallback(@TempDir Path tmp) throws Exception {
    IOException boom = new IOException("simulated network failure");
    ThrowingBackend backend = new ThrowingBackend(boom);

    AtomicLong cbCount = new AtomicLong();
    AtomicReference<Long> cbGen = new AtomicReference<>();
    AtomicReference<Throwable> cbErr = new AtomicReference<>();
    GenerationShippingSubscriber shipper =
        new GenerationShippingSubscriber(
            backend,
            "repl/",
            (genNum, err) -> {
              cbCount.incrementAndGet();
              cbGen.set(genNum);
              cbErr.set(err);
            });

    Path producerRoot = tmp.resolve("producer");
    VectorCollection producer =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.FLAT)
            .storagePath(producerRoot.toAbsolutePath())
            .replicateTo(shipper)
            .build();
    try {
      producer.add(Document.of("a", new float[] {1, 0, 0, 0}));
      producer.commit();

      // Shipping is async — wait for the failure to register on the shipper.
      awaitFailures(shipper, 1, Duration.ofSeconds(5));

      assertThat(shipper.shipFailureCount())
          .as("the put failure must be counted")
          .isGreaterThanOrEqualTo(1L);
      assertThat(shipper.lastShipFailure())
          .as("the most recent failure cause must be retained")
          .isNotNull();
      assertThat(cbCount.get())
          .as("the failure callback must fire at least once")
          .isGreaterThanOrEqualTo(1L);
      assertThat(cbGen.get())
          .as("callback carries the affected generation number")
          .isEqualTo(producer.config().storageRoot() == null ? 0L : 1L);
      assertThat(cbErr.get()).as("callback receives the underlying throwable").isNotNull();
    } finally {
      producer.close();
      shipper.close();
    }
  }

  @Test
  void noFailureKeepsCounterAtZero(@TempDir Path tmp) throws Exception {
    com.integrallis.vectors.storage.backend.HeapStorageBackend backend =
        new com.integrallis.vectors.storage.backend.HeapStorageBackend();
    AtomicLong cbCount = new AtomicLong();
    GenerationShippingSubscriber shipper =
        new GenerationShippingSubscriber(backend, "ok/", (gen, err) -> cbCount.incrementAndGet());

    Path producerRoot = tmp.resolve("producer");
    VectorCollection producer =
        VectorCollection.builder()
            .dimension(4)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.FLAT)
            .storagePath(producerRoot.toAbsolutePath())
            .replicateTo(shipper)
            .build();
    try {
      producer.add(Document.of("a", new float[] {1, 0, 0, 0}));
      producer.commit();
      // Wait until CURRENT is published to the backend (success signal).
      long deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
      while (System.nanoTime() < deadline) {
        if (backend.get("ok/CURRENT") != null) break;
        Thread.sleep(20);
      }
      assertThat(backend.get("ok/CURRENT")).isNotNull();
      assertThat(shipper.shipFailureCount()).isZero();
      assertThat(shipper.lastShipFailure()).isNull();
      assertThat(cbCount.get()).isZero();
    } finally {
      producer.close();
      shipper.close();
    }
  }

  private static void awaitFailures(GenerationShippingSubscriber shipper, long target, Duration max)
      throws InterruptedException {
    long deadline = System.nanoTime() + max.toNanos();
    while (System.nanoTime() < deadline) {
      if (shipper.shipFailureCount() >= target) return;
      Thread.sleep(20);
    }
    throw new AssertionError(
        "shipper.shipFailureCount did not reach " + target + "; got " + shipper.shipFailureCount());
  }

  /** Storage backend whose {@code put} always throws — used to force a shipping failure. */
  private static final class ThrowingBackend implements StorageBackend {
    private final IOException toThrow;

    ThrowingBackend(IOException toThrow) {
      this.toThrow = toThrow;
    }

    @Override
    public void put(String key, byte[] value) throws IOException {
      throw toThrow;
    }

    @Override
    public byte[] get(String key) {
      return null;
    }

    @Override
    public StoredValue getWithEtag(String key) {
      return null;
    }

    @Override
    public byte[] getRange(String key, long offset, int length) {
      return null;
    }

    @Override
    public List<String> list(String prefix) {
      return List.of();
    }

    @Override
    public void delete(String key) {}

    @Override
    public ConditionalPutResult conditionalPut(String key, byte[] value, String expectedEtag) {
      return new ConditionalPutResult(false, null);
    }
  }
}
