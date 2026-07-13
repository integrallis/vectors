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

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.storage.manifest.ManifestGenerationPublisher.GenerationAnnouncer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ManifestGenerationPublisherTest {

  private static final String KEY = "MANIFEST";

  /** Captures announced (generation, hash) pairs — the input a gossip guard would consume. */
  private static final class RecordingAnnouncer implements GenerationAnnouncer {
    final List<long[]> generations = new ArrayList<>();
    final List<String> hashes = new ArrayList<>();

    @Override
    public void announce(long clusterGeneration, String contentHash) {
      generations.add(new long[] {clusterGeneration});
      hashes.add(contentHash);
    }

    long lastGen() {
      return generations.get(generations.size() - 1)[0];
    }
  }

  @Test
  void publishAdvancesManifestAndAnnounces() throws IOException {
    HeapStorageBackend backend = new HeapStorageBackend();
    RecordingAnnouncer announcer = new RecordingAnnouncer();
    var publisher =
        new ManifestGenerationPublisher(new ManifestStore(backend, KEY), "node-a", announcer);

    StorageManifest committed = publisher.publish("collection", 5L, "hash-5", 1_000L);

    assertThat(committed.generation()).isEqualTo(0L); // first cluster generation
    assertThat(committed.entries()).containsEntry("collection", 5L);
    assertThat(committed.contentHash()).isEqualTo("hash-5");
    assertThat(committed.writer()).isEqualTo("node-a");
    assertThat(announcer.generations).hasSize(1);
    assertThat(announcer.lastGen()).isEqualTo(0L);
    assertThat(announcer.hashes).containsExactly("hash-5");
  }

  @Test
  void sequentialPublishesAnnounceMonotonicGenerations() throws IOException {
    HeapStorageBackend backend = new HeapStorageBackend();
    RecordingAnnouncer announcer = new RecordingAnnouncer();
    var publisher =
        new ManifestGenerationPublisher(new ManifestStore(backend, KEY), "node-a", announcer);

    publisher.publish("collection", 5L, "hash-5", 1_000L);
    StorageManifest second = publisher.publish("collection", 6L, "hash-6", 2_000L);

    assertThat(second.generation()).isEqualTo(1L);
    assertThat(second.entries()).containsEntry("collection", 6L); // latest shard generation
    // The announced cluster generations are strictly increasing — exactly what the gossip max-wins
    // guard requires; a guard fed this stream never reloads an older index.
    assertThat(announcer.generations.stream().mapToLong(g -> g[0]).toArray())
        .containsExactly(0L, 1L);
    assertThat(announcer.hashes).containsExactly("hash-5", "hash-6");
  }

  @Test
  void multiShardPublishesAccumulateEntriesAtomically() throws IOException {
    HeapStorageBackend backend = new HeapStorageBackend();
    var publisher =
        new ManifestGenerationPublisher(
            new ManifestStore(backend, KEY), "coordinator", GenerationAnnouncer.NONE);

    publisher.publish("shard-0", 3L, "h0", 1L);
    StorageManifest m = publisher.publish("shard-1", 4L, "h1", 2L);

    // Each shard's generation is recorded; a reader resolves every shard from this one manifest.
    assertThat(m.entries()).containsEntry("shard-0", 3L).containsEntry("shard-1", 4L);
    assertThat(m.generation()).isEqualTo(1L);
    assertThat(new ManifestStore(backend, KEY).load().manifest()).isEqualTo(m);
  }

  @Test
  void announcerNoneIsSafe() throws IOException {
    HeapStorageBackend backend = new HeapStorageBackend();
    var publisher =
        new ManifestGenerationPublisher(
            new ManifestStore(backend, KEY), "node", GenerationAnnouncer.NONE);
    StorageManifest committed = publisher.publish("c", 1L, "h", 0L);
    assertThat(committed.generation()).isEqualTo(0L);
  }
}
