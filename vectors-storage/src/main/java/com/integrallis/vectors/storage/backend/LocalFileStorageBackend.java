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
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;

/**
 * Local-filesystem {@link StorageBackend}. Keys map to files under a root directory; forward
 * slashes in keys become directory separators. Conditional put uses an atomic temp-file rename for
 * crash-safe overwrites.
 *
 * <p>Each file contains a small header with the current etag followed by the raw value bytes.
 * Conditional-put is synchronized on this instance to serialise concurrent writers within a JVM.
 */
public final class LocalFileStorageBackend implements StorageBackend {

  private static final int MAGIC = 0x564c4653; // VLFS
  private static final int VERSION = 1;
  private static final int HEADER_FIXED_BYTES = Integer.BYTES * 3 + Long.BYTES;

  private final Path root;

  public LocalFileStorageBackend(Path root) throws IOException {
    this.root = root.toAbsolutePath().normalize();
    Files.createDirectories(this.root);
  }

  @Override
  public void put(String key, byte[] value) throws IOException {
    Path target = resolve(key);
    Files.createDirectories(target.getParent());
    Path tmp = target.getParent().resolve(target.getFileName() + ".tmp");
    String etag = computeEtag(value);
    writeFramed(tmp, etag, value);
    Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    forceDirectory(target.getParent());
  }

  @Override
  public byte[] get(String key) throws IOException {
    Path target = resolve(key);
    if (!Files.exists(target)) return null;
    return readFramed(target).value();
  }

  @Override
  public byte[] getRange(String key, long offset, int length) throws IOException {
    Path target = resolve(key);
    if (!Files.exists(target)) return null;
    try (FileChannel ch = FileChannel.open(target, StandardOpenOption.READ)) {
      FramedHeader header = readFramedHeader(ch, target);
      long valueOffset = header == null ? 0 : header.valueOffset();
      long size = header == null ? ch.size() : header.valueLength();
      if (offset < 0 || length < 0 || offset + length > size) {
        throw new IndexOutOfBoundsException(
            "getRange(" + key + ", offset=" + offset + ", length=" + length + ") size=" + size);
      }
      if (length == 0) return new byte[0];
      byte[] out = new byte[length];
      readFullyAt(ch, ByteBuffer.wrap(out), valueOffset + offset, key);
      return out;
    }
  }

  @Override
  public List<String> list(String prefix) throws IOException {
    List<String> result = new ArrayList<>();
    if (!Files.exists(root)) return result;
    try (var stream = Files.walk(root)) {
      stream
          .filter(Files::isRegularFile)
          .map(p -> root.relativize(p).toString().replace(java.io.File.separatorChar, '/'))
          .filter(k -> !k.endsWith(".tmp") && k.startsWith(prefix))
          .forEach(result::add);
    }
    return result;
  }

  @Override
  public void delete(String key) throws IOException {
    Path target = resolve(key);
    Files.deleteIfExists(target);
    Files.deleteIfExists(legacyEtagPath(key));
    forceDirectory(target.getParent());
  }

  @Override
  public synchronized ConditionalPutResult conditionalPut(
      String key, byte[] value, String expectedEtag) throws IOException {
    String currentEtag = readEtag(key);
    if (!etagEquals(currentEtag, expectedEtag)) {
      return new ConditionalPutResult(false, null);
    }
    String newEtag = computeEtag(value);
    put(key, value);
    return new ConditionalPutResult(true, newEtag);
  }

  // --- internals ---

  private Path resolve(String key) throws IOException {
    if (key == null || key.isEmpty()) {
      throw new IOException("storage key must be non-empty");
    }
    Path keyPath = Path.of(key.replace('/', java.io.File.separatorChar));
    if (keyPath.isAbsolute()) {
      throw new IOException("storage key must be relative: " + key);
    }
    for (Path part : keyPath) {
      if ("..".equals(part.toString())) {
        throw new IOException("storage key must not contain '..': " + key);
      }
    }
    Path resolved = root.resolve(keyPath).normalize();
    if (!resolved.startsWith(root)) {
      throw new IOException("storage key escapes root: " + key);
    }
    return resolved;
  }

  private Path legacyEtagPath(String key) throws IOException {
    return resolve(key + ".etag");
  }

  private String readEtag(String key) throws IOException {
    Path p = resolve(key);
    if (!Files.exists(p)) return null;
    return readFramed(p).etag();
  }

  private static void writeFramed(Path file, String etag, byte[] value) throws IOException {
    byte[] etagBytes = etag.getBytes(java.nio.charset.StandardCharsets.UTF_8);
    ByteBuffer header = ByteBuffer.allocate(HEADER_FIXED_BYTES + etagBytes.length);
    header.putInt(MAGIC);
    header.putInt(VERSION);
    header.putInt(etagBytes.length);
    header.putLong(value.length);
    header.put(etagBytes);
    header.flip();
    try (FileChannel ch =
        FileChannel.open(
            file,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
            StandardOpenOption.WRITE)) {
      writeFully(ch, header);
      writeFully(ch, ByteBuffer.wrap(value));
      ch.force(true);
    }
  }

  private static FramedValue readFramed(Path file) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    if (bytes.length < HEADER_FIXED_BYTES) {
      return readLegacy(file, bytes);
    }
    ByteBuffer buf = ByteBuffer.wrap(bytes);
    int magic = buf.getInt();
    if (magic != MAGIC) {
      return readLegacy(file, bytes);
    }
    int version = buf.getInt();
    if (version != VERSION) {
      throw new IOException("unsupported local storage file version: " + version);
    }
    int etagLength = buf.getInt();
    long valueLength = buf.getLong();
    if (etagLength < 0 || valueLength < 0 || valueLength > Integer.MAX_VALUE) {
      throw new IOException("invalid local storage header in " + file);
    }
    long expectedLength = HEADER_FIXED_BYTES + (long) etagLength + valueLength;
    if (expectedLength != bytes.length) {
      throw new IOException(
          "local storage length mismatch for "
              + file
              + ": expected "
              + expectedLength
              + " bytes, got "
              + bytes.length);
    }
    byte[] etagBytes = new byte[etagLength];
    buf.get(etagBytes);
    byte[] value = new byte[(int) valueLength];
    buf.get(value);
    return new FramedValue(new String(etagBytes, java.nio.charset.StandardCharsets.UTF_8), value);
  }

  private static FramedHeader readFramedHeader(FileChannel ch, Path file) throws IOException {
    if (ch.size() < HEADER_FIXED_BYTES) {
      return null;
    }
    ByteBuffer header = ByteBuffer.allocate(HEADER_FIXED_BYTES);
    readFullyAt(ch, header, 0, file.toString());
    header.flip();
    int magic = header.getInt();
    if (magic != MAGIC) {
      return null;
    }
    int version = header.getInt();
    if (version != VERSION) {
      throw new IOException("unsupported local storage file version: " + version);
    }
    int etagLength = header.getInt();
    long valueLength = header.getLong();
    if (etagLength < 0 || valueLength < 0) {
      throw new IOException("invalid local storage header in " + file);
    }
    long valueOffset = HEADER_FIXED_BYTES + (long) etagLength;
    long expectedLength = valueOffset + valueLength;
    if (expectedLength != ch.size()) {
      throw new IOException(
          "local storage length mismatch for "
              + file
              + ": expected "
              + expectedLength
              + " bytes, got "
              + ch.size());
    }
    return new FramedHeader(valueOffset, valueLength);
  }

  private static FramedValue readLegacy(Path file, byte[] value) throws IOException {
    Path etagFile = file.resolveSibling(file.getFileName() + ".etag");
    String etag = Files.exists(etagFile) ? Files.readString(etagFile).trim() : computeEtag(value);
    return new FramedValue(etag, value);
  }

  private static void writeFully(FileChannel ch, ByteBuffer buffer) throws IOException {
    while (buffer.hasRemaining()) {
      ch.write(buffer);
    }
  }

  private static void readFullyAt(FileChannel ch, ByteBuffer buffer, long offset, String key)
      throws IOException {
    int total = 0;
    while (buffer.hasRemaining()) {
      int n = ch.read(buffer, offset + total);
      if (n < 0) {
        throw new IOException("unexpected EOF reading " + key + " at offset " + (offset + total));
      }
      total += n;
    }
  }

  private static void forceDirectory(Path dir) throws IOException {
    try (FileChannel ch = FileChannel.open(dir, StandardOpenOption.READ)) {
      ch.force(true);
    } catch (UnsupportedOperationException | UncheckedIOException ignored) {
      // Some filesystems do not allow opening directories as channels. The value file itself has
      // already been forced; directory fsync is best-effort on those platforms.
    }
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

  private record FramedValue(String etag, byte[] value) {}

  private record FramedHeader(long valueOffset, long valueLength) {}
}
