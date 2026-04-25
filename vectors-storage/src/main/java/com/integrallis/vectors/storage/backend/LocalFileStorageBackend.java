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
package com.integrallis.vectors.storage.backend;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;

/**
 * Local-filesystem {@link StorageBackend}. Keys map to files under a root directory; forward
 * slashes in keys become directory separators. Conditional put uses an atomic temp-file rename for
 * crash-safe overwrites.
 *
 * <p>Each value file has a companion {@code <key>.etag} sidecar that stores the current etag.
 * Conditional-put is synchronized on this instance to serialise concurrent writers within a JVM.
 */
public final class LocalFileStorageBackend implements StorageBackend {

  private final Path root;

  public LocalFileStorageBackend(Path root) throws IOException {
    this.root = root;
    Files.createDirectories(root);
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    Path target = resolve(key);
    Files.createDirectories(target.getParent());
    Path tmp = target.getParent().resolve(target.getFileName() + ".tmp");
    Files.write(tmp, value);
    Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    writeEtag(key, computeEtag(value));
  }

  @Override
  public byte[] get(String key) throws IOException {
    Path target = resolve(key);
    if (!Files.exists(target)) return null;
    return Files.readAllBytes(target);
  }

  @Override
  public List<String> list(String prefix) throws IOException {
    List<String> result = new ArrayList<>();
    if (!Files.exists(root)) return result;
    try (var stream = Files.walk(root)) {
      stream
          .filter(Files::isRegularFile)
          .map(p -> root.relativize(p).toString().replace(java.io.File.separatorChar, '/'))
          .filter(k -> !k.endsWith(".etag") && k.startsWith(prefix))
          .forEach(result::add);
    }
    return result;
  }

  @Override
  public void delete(String key) throws IOException {
    Files.deleteIfExists(resolve(key));
    Files.deleteIfExists(etagPath(key));
  }

  @Override
  public synchronized ConditionalPutResult conditionalPut(
      String key, byte[] value, String expectedEtag) throws IOException {
    String currentEtag = readEtag(key);
    if (!etagEquals(currentEtag, expectedEtag)) {
      return new ConditionalPutResult(false, null);
    }
    String newEtag = computeEtag(value);
    put(key, value); // atomic move — also writes etag sidecar
    return new ConditionalPutResult(true, newEtag);
  }

  // --- internals ---

  private Path resolve(String key) {
    return root.resolve(key.replace('/', java.io.File.separatorChar));
  }

  private Path etagPath(String key) {
    return resolve(key + ".etag");
  }

  private String readEtag(String key) throws IOException {
    Path p = etagPath(key);
    if (!Files.exists(p)) return null;
    return Files.readString(p).trim();
  }

  private void writeEtag(String key, String etag) throws IOException {
    Path p = etagPath(key);
    Files.createDirectories(p.getParent());
    Files.writeString(p, etag);
  }

  private static boolean etagEquals(String current, String expected) {
    if (expected == null) return current == null;
    return expected.equals(current);
  }

  private static String computeEtag(byte[] value) {
    CRC32 crc = new CRC32();
    crc.update(value);
    return Long.toHexString(crc.getValue()) + "-" + value.length;
  }
}
