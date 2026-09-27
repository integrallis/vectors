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

import com.integrallis.vectors.db.storage.FileFormat;
import com.integrallis.vectors.db.storage.Manifest;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Reads the recipe anchor out of a persisted collection without opening it.
 *
 * <p>Needed because the recipe has to be reconciled <em>before</em> the collection is built — the
 * config it produces is an input to construction — so the anchor must be readable from the manifest
 * on disk alone.
 */
final class StoredRecipeAnchor {

  private StoredRecipeAnchor() {}

  /** The recipe hash recorded in the newest committed generation's manifest, if any. */
  static Optional<String> read(Path collectionRoot) throws IOException {
    Optional<Path> manifest = newestManifest(collectionRoot);
    if (manifest.isEmpty()) {
      return Optional.empty();
    }
    try {
      return Manifest.fromBytes(Files.readAllBytes(manifest.get())).recipeHash();
    } catch (IOException unreadable) {
      // A manifest this build cannot parse is not this method's problem to report -- the collection
      // open path will fail with a far better message. Treat it as "no anchor" and let that happen.
      return Optional.empty();
    }
  }

  /** Whether the collection already holds committed vectors. */
  static boolean hasVectors(Path collectionRoot) throws IOException {
    Optional<Path> manifest = newestManifest(collectionRoot);
    if (manifest.isEmpty()) {
      return false;
    }
    try {
      return Manifest.fromBytes(Files.readAllBytes(manifest.get())).liveCount() > 0L;
    } catch (IOException unreadable) {
      return false;
    }
  }

  private static Optional<Path> newestManifest(Path collectionRoot) throws IOException {
    if (!Files.isDirectory(collectionRoot)) {
      return Optional.empty();
    }
    try (Stream<Path> generations = Files.list(collectionRoot)) {
      return generations
          .filter(Files::isDirectory)
          .filter(p -> p.getFileName().toString().startsWith(FileFormat.GENERATION_DIR_PREFIX))
          .max(Comparator.comparing(p -> p.getFileName().toString()))
          .map(p -> p.resolve(FileFormat.MANIFEST_FILE))
          .filter(Files::isRegularFile);
    }
  }
}
