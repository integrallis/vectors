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
package com.integrallis.vectors.storage.manifest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ManifestStoreTest {

  private static final String KEY = "MANIFEST";

  @Test
  void loadOnEmptyBackendReturnsEmptyWithNullEtag() throws IOException {
    ManifestStore store = new ManifestStore(new HeapStorageBackend(), KEY);
    ManifestStore.Loaded loaded = store.load();
    assertThat(loaded.manifest().isEmpty()).isTrue();
    assertThat(loaded.etag()).isNull();
  }

  @Test
  void firstCommitStartsAtGenerationZeroAndPersists() throws IOException {
    HeapStorageBackend backend = new HeapStorageBackend();
    ManifestStore store = new ManifestStore(backend, KEY);

    StorageManifest committed =
        store.commit(cur -> cur.withEntry("shard-0", 5L).withProvenance("h0", 1L, "w"));

    assertThat(committed.generation()).isEqualTo(0L);
    assertThat(committed.entries()).containsEntry("shard-0", 5L);
    // A fresh store reading only durable state agrees.
    assertThat(new ManifestStore(backend, KEY).load().manifest()).isEqualTo(committed);
  }

  @Test
  void sequentialCommitsAdvanceMonotonically() throws IOException {
    HeapStorageBackend backend = new HeapStorageBackend();
    ManifestStore store = new ManifestStore(backend, KEY);

    StorageManifest g0 = store.commit(cur -> cur.withEntry("a", 1L).withProvenance("h0", 0L, "w"));
    StorageManifest g1 = store.commit(cur -> cur.withEntry("b", 2L).withProvenance("h1", 0L, "w"));

    assertThat(g0.generation()).isEqualTo(0L);
    assertThat(g1.generation()).isEqualTo(1L);
    // Second commit rebased onto the first: both entries survive.
    assertThat(g1.entries()).containsEntry("a", 1L).containsEntry("b", 2L);
  }

  @Test
  void casConflictRebasesAndRetries() throws IOException {
    HeapStorageBackend backend = new HeapStorageBackend();
    ManifestStore store = new ManifestStore(backend, KEY);
    AtomicInteger applyCount = new AtomicInteger();

    StorageManifest result =
        store.commit(
            cur -> {
              // On the first apply, a concurrent writer lands between our load() and CAS: advance
              // the manifest out-of-band. Our CAS then loses and commit() must re-read + rebase.
              if (applyCount.getAndIncrement() == 0) {
                new ManifestStore(backend, KEY)
                    .commit(c -> c.withEntry("other", 9L).withProvenance("h-other", 0L, "w2"));
              }
              return cur.withEntry("mine", 7L).withProvenance("h-mine", 0L, "w1");
            });

    assertThat(applyCount.get()).as("delta re-ran once after the conflict").isEqualTo(2);
    assertThat(result.generation()).isEqualTo(1L); // gen 0 = the concurrent writer, gen 1 = ours
    assertThat(result.entries())
        .as("rebase preserved the concurrent writer's entry AND added ours")
        .containsEntry("other", 9L)
        .containsEntry("mine", 7L);
  }

  @Test
  void concurrentCommitsAllLandWithNoLostUpdates() throws Exception {
    HeapStorageBackend backend = new HeapStorageBackend();
    int writers = 12;
    var pool = java.util.concurrent.Executors.newFixedThreadPool(writers);
    try {
      var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
      for (int i = 0; i < writers; i++) {
        final String k = "shard-" + i;
        final long v = i;
        futures.add(
            pool.submit(
                () -> {
                  try {
                    new ManifestStore(backend, KEY)
                        .commit(cur -> cur.withEntry(k, v).withProvenance("h-" + k, 0L, "w"), 200);
                  } catch (IOException e) {
                    throw new RuntimeException(e);
                  }
                }));
      }
      for (var f : futures) f.get();
    } finally {
      pool.shutdownNow();
    }

    StorageManifest finalManifest = new ManifestStore(backend, KEY).load().manifest();
    assertThat(finalManifest.entries()).as("every writer's entry survived").hasSize(writers);
    // Gap-free monotonic: N commits over an empty start end at generation N-1.
    assertThat(finalManifest.generation()).isEqualTo((long) writers - 1);
  }

  @Test
  void persistentContentionThrowsManifestConflict() {
    HeapStorageBackend backend = new HeapStorageBackend();
    ManifestStore store = new ManifestStore(backend, KEY);

    // A delta that advances the manifest out-of-band on EVERY apply so our CAS can never win.
    assertThatThrownBy(
            () ->
                store.commit(
                    cur -> {
                      new ManifestStore(backend, KEY)
                          .commit(c -> c.withProvenance("bump", 0L, "w2"), 50);
                      return cur.withEntry("mine", 1L).withProvenance("h", 0L, "w1");
                    },
                    3))
        .isInstanceOf(ManifestConflictException.class)
        .hasMessageContaining("3 attempts");
  }
}
