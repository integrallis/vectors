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
import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.backend.LocalFileStorageBackend;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end P3.1 read-replica directory shipping: a producer commits, a {@link
 * GenerationShippingSubscriber} ships the generation to a {@link StorageBackend}, and a follower
 * picks it up (via {@link GenerationSync#pull} or directly) and serves it after {@link
 * VectorCollection#refresh()}.
 */
@Tag("unit")
class GenerationShippingTest {

  private static final int DIM = 8;

  private static float[] vec(float base) {
    float[] v = new float[DIM];
    for (int i = 0; i < DIM; i++) {
      v[i] = base + i * 0.01f;
    }
    return v;
  }

  private static VectorCollection flatPersistent(Path root) {
    return VectorCollection.builder()
        .dimension(DIM)
        .metric(SimilarityFunction.EUCLIDEAN)
        .indexType(IndexType.FLAT)
        .storagePath(root)
        .build();
  }

  @Test
  void heapShipThenPullThenRefreshServesGeneration(@TempDir Path tmp) throws Exception {
    Path producerRoot = tmp.resolve("producer");
    Path followerRoot = tmp.resolve("follower");
    HeapStorageBackend backend = new HeapStorageBackend();
    GenerationShippingSubscriber shipper = new GenerationShippingSubscriber(backend, "repl/");

    VectorCollection producer =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.FLAT)
            .storagePath(producerRoot)
            .replicateTo(shipper)
            .build();
    producer.add(Document.of("a", vec(1f)));
    producer.add(Document.of("b", vec(2f)));
    producer.commit();

    // Shipping is asynchronous; wait for the remote CURRENT to publish.
    waitFor(() -> backend.get("repl/CURRENT") != null, "remote CURRENT");

    VectorCollection follower = flatPersistent(followerRoot);
    assertThat(follower.size()).isZero();

    long pulled = GenerationSync.pull(backend, "repl/", followerRoot);
    assertThat(pulled).isGreaterThanOrEqualTo(0L);
    assertThat(follower.refresh()).isTrue();
    assertThat(follower.size()).isEqualTo(2);
    assertThat(follower.search(SearchRequest.builder(vec(1f), 1).build()).hits().get(0).id())
        .isEqualTo("a");

    // Second pull with nothing new shipped is a no-op.
    assertThat(follower.refresh()).isFalse();

    producer.close();
    follower.close();
    shipper.close();
  }

  @Test
  void localBackendIncrementalPullRefresh(@TempDir Path tmp) throws Exception {
    Path producerRoot = tmp.resolve("producer");
    Path transport = tmp.resolve("transport"); // the replication "blob store" (opaque framing)
    Path followerRoot = tmp.resolve("follower");
    LocalFileStorageBackend backend = new LocalFileStorageBackend(transport);
    GenerationShippingSubscriber shipper = new GenerationShippingSubscriber(backend, "");

    VectorCollection producer =
        VectorCollection.builder()
            .dimension(DIM)
            .metric(SimilarityFunction.EUCLIDEAN)
            .indexType(IndexType.FLAT)
            .storagePath(producerRoot)
            .replicateTo(shipper)
            .build();
    VectorCollection follower = flatPersistent(followerRoot);

    producer.add(Document.of("a", vec(1f)));
    producer.commit();
    waitFor(() -> remoteCurrent(backend) >= 0L, "first shipped generation");
    long g1 = GenerationSync.pull(backend, "", followerRoot);
    assertThat(g1).isGreaterThanOrEqualTo(0L);
    assertThat(follower.refresh()).isTrue();
    assertThat(follower.size()).isEqualTo(1);

    // A new generation is committed and shipped.
    producer.add(Document.of("b", vec(2f)));
    producer.commit();
    waitFor(() -> remoteCurrent(backend) > g1, "second shipped generation");
    long g2 = GenerationSync.pull(backend, "", followerRoot);
    assertThat(g2).isGreaterThan(g1);

    // The follower keeps serving the old generation until it refreshes.
    assertThat(follower.size()).isEqualTo(1);
    assertThat(follower.refresh()).isTrue();
    assertThat(follower.size()).isEqualTo(2);
    assertThat(follower.search(SearchRequest.builder(vec(2f), 1).build()).hits().get(0).id())
        .isEqualTo("b");

    producer.close();
    follower.close();
    shipper.close();
  }

  /** Reads the shipped {@code CURRENT} via the backend (which unframes), or -1 if absent. */
  private static long remoteCurrent(StorageBackend backend) {
    try {
      byte[] bytes = backend.get("CURRENT");
      if (bytes == null || bytes.length < Long.BYTES) {
        return -1L;
      }
      return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
    } catch (Exception e) {
      return -1L;
    }
  }

  private interface Condition {
    boolean met() throws Exception;
  }

  private static void waitFor(Condition condition, String what) throws Exception {
    long deadline = System.nanoTime() + 10_000_000_000L; // 10s
    while (System.nanoTime() < deadline) {
      if (condition.met()) {
        return;
      }
      Thread.sleep(20);
    }
    throw new AssertionError("timed out waiting for " + what);
  }
}
