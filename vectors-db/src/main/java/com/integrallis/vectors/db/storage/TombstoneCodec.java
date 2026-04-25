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
package com.integrallis.vectors.db.storage;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.BitSet;

/**
 * Encodes and decodes a {@link BitSet} of tombstoned ordinals to/from the {@code tombstones.bin}
 * on-disk format.
 *
 * <p>Layout (little-endian throughout):
 *
 * <pre>
 * Offset  Size  Field
 *   0      4    magic (FileFormat.MAGIC_TOMBSTONES = "VTBS")
 *   4      4    version (FileFormat.VERSION_TOMBSTONES = 1)
 *   8      4    physicalCount (total ordinals in the generation)
 *  12      4    reserved (must be 0)
 *  16      N*8  long[] words from BitSet.toLongArray(), little-endian
 * </pre>
 *
 * <p>The header is 16 bytes. The body is {@code ceil(physicalCount / 64) * 8} bytes at most, but
 * {@link BitSet#toLongArray()} only emits up to the highest set bit — trailing zero words are
 * omitted. The decoder uses {@code BitSet.valueOf(long[])} which tolerates a shorter array.
 *
 * <p>If no tombstones exist, the file is not written (tombstonesBinLength == 0 in the manifest).
 */
public final class TombstoneCodec {

  /** Fixed header size: magic(4) + version(4) + physicalCount(4) + reserved(4). */
  public static final int HEADER_SIZE = 16;

  private TombstoneCodec() {}

  /**
   * Encodes a tombstone BitSet into bytes suitable for writing to {@code tombstones.bin}.
   *
   * @param tombstones the tombstone bitset (must not be null)
   * @param physicalCount total ordinals in the generation (for validation on decode)
   * @return encoded bytes, or a zero-length array if the bitset is empty
   */
  public static byte[] encode(BitSet tombstones, int physicalCount) {
    if (tombstones.isEmpty()) {
      return new byte[0];
    }
    long[] words = tombstones.toLongArray();
    int bodyBytes = words.length * Long.BYTES;
    byte[] out = new byte[HEADER_SIZE + bodyBytes];
    ByteBuffer buf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(FileFormat.MAGIC_TOMBSTONES);
    buf.putInt(FileFormat.VERSION_TOMBSTONES);
    buf.putInt(physicalCount);
    buf.putInt(0); // reserved
    for (long w : words) {
      buf.putLong(w);
    }
    return out;
  }

  /**
   * Decodes a tombstone BitSet from bytes read from {@code tombstones.bin}.
   *
   * @param bytes the raw file contents
   * @return the decoded BitSet
   * @throws IOException if the magic, version, or layout is invalid
   */
  public static BitSet decode(byte[] bytes) throws IOException {
    if (bytes == null || bytes.length == 0) {
      return new BitSet();
    }
    if (bytes.length < HEADER_SIZE) {
      throw new IOException(
          "tombstones.bin truncated: expected at least "
              + HEADER_SIZE
              + " bytes, got "
              + bytes.length);
    }
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    int magic = buf.getInt();
    if (magic != FileFormat.MAGIC_TOMBSTONES) {
      throw new IOException(
          String.format(
              "tombstones.bin magic mismatch: expected 0x%08x, got 0x%08x",
              FileFormat.MAGIC_TOMBSTONES, magic));
    }
    int version = buf.getInt();
    if (version != FileFormat.VERSION_TOMBSTONES) {
      throw new IOException(
          "tombstones.bin version mismatch: expected "
              + FileFormat.VERSION_TOMBSTONES
              + ", got "
              + version);
    }
    int physicalCount = buf.getInt();
    int reserved = buf.getInt();
    if (reserved != 0) {
      throw new IOException("tombstones.bin reserved field must be 0, got " + reserved);
    }

    int bodyBytes = bytes.length - HEADER_SIZE;
    if (bodyBytes % Long.BYTES != 0) {
      throw new IOException(
          "tombstones.bin body size " + bodyBytes + " is not a multiple of 8 bytes");
    }
    int wordCount = bodyBytes / Long.BYTES;
    long[] words = new long[wordCount];
    for (int i = 0; i < wordCount; i++) {
      words[i] = buf.getLong();
    }
    BitSet tombstones = BitSet.valueOf(words);
    // Validate that no tombstone bit exceeds the physical ordinal range. A bit at or above
    // physicalCount indicates file corruption (e.g. a truncated or mismatched tombstones.bin).
    // If this slips through, liveCount() would under-count live documents silently.
    if (tombstones.length() > physicalCount) {
      throw new IOException(
          "tombstones.bin highest set bit "
              + (tombstones.length() - 1)
              + " is out of range for physicalCount "
              + physicalCount);
    }
    return tombstones;
  }
}
