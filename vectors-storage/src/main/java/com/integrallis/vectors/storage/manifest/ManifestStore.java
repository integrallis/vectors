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

import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.util.Objects;

/**
 * Reads and atomically advances a {@link StorageManifest} on a {@link StorageBackend} using the
 * backend's conditional-put (compare-and-swap) primitive. This is the object-storage commit point:
 * the manifest is the single object whose CAS makes a new generation live for every reader at once.
 *
 * <p>The write path is optimistic-concurrency, exactly as object-storage table formats do it
 * (Iceberg / Delta / Lance / TurboPuffer): read the current manifest with its etag, apply the
 * caller's delta, stamp {@code generation = current + 1}, and {@code conditionalPut(…, etag)}. If a
 * concurrent writer advanced the manifest first the CAS fails; {@link #commit} then re-reads and
 * rebases the delta onto the newer manifest and retries, so the generation stays strictly monotonic
 * and gap-free and no committed delta is lost.
 *
 * <p>Content payloads (the {@code gen-N/} objects a delta refers to) must be written durably
 * <em>before</em> {@code commit} — they are content-addressed and never overwrite, so a lost CAS
 * simply leaves them unreferenced for a later GC sweep; only the manifest swap publishes them.
 */
public final class ManifestStore {

  /** Default CAS attempt cap before giving up with a {@link ManifestConflictException}. */
  public static final int DEFAULT_MAX_ATTEMPTS = 16;

  private final StorageBackend backend;
  private final String key;

  public ManifestStore(StorageBackend backend, String key) {
    this.backend = Objects.requireNonNull(backend, "backend");
    if (key == null || key.isBlank()) {
      throw new IllegalArgumentException("manifest key must be non-blank");
    }
    this.key = key;
  }

  /** The manifest key this store reads/writes. */
  public String key() {
    return key;
  }

  /** A loaded manifest together with the etag to CAS against ({@code null} etag = absent). */
  public record Loaded(StorageManifest manifest, String etag) {}

  /**
   * Reads the current manifest and its etag. Returns {@code (EMPTY, null)} when no manifest object
   * exists yet, so a first {@link #commit} lands with an {@code expectedEtag} of {@code null}.
   */
  public Loaded load() throws IOException {
    StorageBackend.StoredValue stored = backend.getWithEtag(key);
    if (stored == null) {
      return new Loaded(StorageManifest.EMPTY, null);
    }
    return new Loaded(StorageManifest.decode(stored.value()), stored.etag());
  }

  /**
   * Attempts a single CAS of {@code next} against {@code expectedEtag} ({@code null} = expect
   * absent). Returns the new etag on success, or {@code null} if a concurrent writer won the race.
   */
  public String compareAndPut(StorageManifest next, String expectedEtag) throws IOException {
    Objects.requireNonNull(next, "next");
    StorageBackend.ConditionalPutResult r =
        backend.conditionalPut(key, next.encode(), expectedEtag);
    return r.succeeded() ? r.newEtag() : null;
  }

  /**
   * Delta applied to the current manifest to produce the new committed content (entries + hash).
   */
  @FunctionalInterface
  public interface ManifestDelta {
    /**
     * Returns the new manifest content given the current one. The returned generation is ignored —
     * {@link #commit} stamps {@code current.generation() + 1}. Rerun on each CAS retry, so it must
     * be a pure function of {@code current} (a "rebase").
     */
    StorageManifest apply(StorageManifest current) throws IOException;
  }

  /** Convenience: {@link #commit(ManifestDelta, int)} with {@link #DEFAULT_MAX_ATTEMPTS}. */
  public StorageManifest commit(ManifestDelta delta) throws IOException {
    return commit(delta, DEFAULT_MAX_ATTEMPTS);
  }

  /**
   * Read → apply delta → stamp {@code generation+1} → CAS, retrying on conflict up to {@code
   * maxAttempts}. Returns the committed manifest (the newly published generation).
   *
   * @throws ManifestConflictException if every attempt loses the CAS race (persistent contention)
   */
  public StorageManifest commit(ManifestDelta delta, int maxAttempts) throws IOException {
    Objects.requireNonNull(delta, "delta");
    if (maxAttempts <= 0) {
      throw new IllegalArgumentException("maxAttempts must be > 0: " + maxAttempts);
    }
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      Loaded current = load();
      StorageManifest proposed = delta.apply(current.manifest());
      // Monotonic, gap-free: the new generation is always one past whatever we just read, even
      // after
      // a rebase against a concurrently-advanced manifest.
      StorageManifest next = proposed.atGeneration(nextGeneration(current.manifest()));
      String newEtag = compareAndPut(next, current.etag());
      if (newEtag != null) {
        return next;
      }
      // Lost the race — another writer advanced the manifest. Loop: re-read, rebase, retry.
    }
    throw new ManifestConflictException(
        "manifest CAS lost after " + maxAttempts + " attempts on key '" + key + "'");
  }

  private static long nextGeneration(StorageManifest current) {
    return current.isEmpty() ? 0L : current.generation() + 1L;
  }
}
