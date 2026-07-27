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

import java.io.IOException;
import java.util.Objects;

/**
 * Publishes a committed generation through the object-storage manifest and announces it — the
 * write-side glue for phase-4 (see {@code
 * java-ai/vectors-phase4-manifest-cas-design-2026-07-13.md}).
 *
 * <p>A commit path (the generation shipper for a single-shard collection, or the cluster committer
 * for a sharded one) calls {@link #publish} once the generation's payload objects are durably
 * written. {@code publish} sets {@code entryId → generation} in the manifest via {@link
 * ManifestStore#commit} (a monotonic, race-safe CAS), then hands the resulting <em>cluster</em>
 * generation and content hash to a {@link GenerationAnnouncer}. Wiring the announcer to {@code
 * GossipClusterMembership.announceVersion(long, String)} closes the loop: the manifest is the
 * source of the monotonic generation the gossip guard already rejects stale announcements against,
 * so a reordered publish can never tell followers to reload an older index.
 *
 * <p>The announcer is a plain callback so this storage-layer class stays decoupled from the
 * distributed membership module; the caller supplies the bridge lambda.
 */
public final class ManifestGenerationPublisher {

  /** Receives the cluster generation and content hash of a freshly-published manifest. */
  @FunctionalInterface
  public interface GenerationAnnouncer {
    /**
     * @param clusterGeneration the manifest's new (monotonic) generation
     * @param contentHash the published content hash
     */
    void announce(long clusterGeneration, String contentHash);

    /** An announcer that does nothing (single-node / no gossip). */
    GenerationAnnouncer NONE = (gen, hash) -> {};
  }

  private final ManifestStore store;
  private final String writer;
  private final GenerationAnnouncer announcer;

  /**
   * @param store the manifest store (backend + key)
   * @param writer provenance tag recorded in the manifest (e.g. node id)
   * @param announcer invoked with {@code (clusterGeneration, contentHash)} after each successful
   *     publish; use {@link GenerationAnnouncer#NONE} for no gossip
   */
  public ManifestGenerationPublisher(
      ManifestStore store, String writer, GenerationAnnouncer announcer) {
    this.store = Objects.requireNonNull(store, "store");
    this.writer = Objects.requireNonNull(writer, "writer");
    this.announcer = Objects.requireNonNull(announcer, "announcer");
  }

  /**
   * Records {@code entryId → generation} with {@code contentHash} in the manifest via CAS, then
   * announces the resulting cluster generation and hash. Returns the committed manifest.
   *
   * @param entryId the manifest entry to advance (the shard id, or the collection id for a
   *     single-shard collection)
   * @param generation this entry's committed generation
   * @param contentHash opaque identity of the committed state (e.g. a hash of the generation
   *     payload)
   * @param committedAtEpochMs commit timestamp recorded in the manifest
   * @throws ManifestConflictException if the manifest CAS loses under persistent contention
   */
  public StorageManifest publish(
      String entryId, long generation, String contentHash, long committedAtEpochMs)
      throws IOException {
    Objects.requireNonNull(entryId, "entryId");
    Objects.requireNonNull(contentHash, "contentHash");
    StorageManifest committed =
        store.commit(
            current ->
                current
                    .withEntry(entryId, generation)
                    .withProvenance(contentHash, committedAtEpochMs, writer));
    announcer.announce(committed.generation(), committed.contentHash());
    return committed;
  }
}
