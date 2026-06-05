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
import com.integrallis.vectors.db.storage.GenerationDirectory;
import com.integrallis.vectors.storage.backend.StorageBackend;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Follower-side counterpart to {@link GenerationShippingSubscriber}: downloads the latest shipped
 * generation from a {@link StorageBackend} into a local storage root, so a replica that serves
 * reads via mmap (which needs local files) can {@link VectorCollection#refresh()} onto it (P3.1).
 *
 * <p>For a {@code file://} target a replica can open the shipped directory directly and skip this
 * step; {@code pull} exists for {@code s3://} (and other remote) backends where the bytes must be
 * materialised locally first.
 */
public final class GenerationSync {

  private GenerationSync() {}

  /**
   * Downloads the generation referenced by the remote {@code CURRENT} into {@code localRoot}, then
   * publishes a local {@code CURRENT} pointing at it (written last, atomically). After a successful
   * pull, call {@link VectorCollection#refresh()} on a collection opened at {@code localRoot} to
   * serve the new generation.
   *
   * <p>Idempotent and safe to call repeatedly (e.g. on a poll loop): files are overwritten and the
   * local {@code CURRENT} is only advanced once all payload files are present.
   *
   * @param backend the source store (the shipping target)
   * @param keyPrefix the same prefix the shipper used
   * @param localRoot the replica's local storage root
   * @return the pulled generation number, or {@code -1} if nothing has been shipped yet
   * @throws IOException on storage or filesystem failure
   */
  public static long pull(StorageBackend backend, String keyPrefix, Path localRoot)
      throws IOException {
    String prefix = normalizePrefix(keyPrefix);

    byte[] currentBytes = backend.get(prefix + FileFormat.CURRENT_FILE);
    if (currentBytes == null || currentBytes.length < Long.BYTES) {
      return -1L; // nothing shipped yet
    }
    long generationNumber = ByteBuffer.wrap(currentBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();

    String genDirName = FileFormat.generationDirName(generationNumber);
    String genKeyPrefix = prefix + genDirName + "/";
    List<String> keys = backend.list(genKeyPrefix);
    if (keys.isEmpty()) {
      return -1L; // CURRENT present but payload not (yet) listable — treat as not ready
    }

    Path localGenDir = localRoot.resolve(genDirName);
    Files.createDirectories(localGenDir);
    for (String key : keys) {
      byte[] data = backend.get(key);
      if (data == null) {
        continue;
      }
      String fileName = key.substring(key.lastIndexOf('/') + 1);
      Files.write(localGenDir.resolve(fileName), data);
    }

    // Publish the local CURRENT last (atomic rename + fsync), mirroring the commit discipline.
    GenerationDirectory.writeCurrentAtomic(localRoot, generationNumber);
    return generationNumber;
  }

  private static String normalizePrefix(String prefix) {
    if (prefix == null || prefix.isEmpty() || prefix.equals("/")) {
      return "";
    }
    String p = prefix.startsWith("/") ? prefix.substring(1) : prefix;
    return p.endsWith("/") ? p : p + "/";
  }
}
