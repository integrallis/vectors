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
package com.integrallis.vectors.storage.io;

import java.io.IOException;
import java.lang.foreign.MemorySegment;

/**
 * Random-access output for writing on-disk data structures. Stateful (has current position) and NOT
 * thread-safe.
 *
 * <p>All multi-byte writes use little-endian byte order.
 */
public interface RandomAccessOutput extends AutoCloseable {

  /** Sets the current write position. */
  void seek(long offset) throws IOException;

  /** Returns the current write position. */
  long position() throws IOException;

  /** Writes a 32-bit little-endian int and advances by 4 bytes. */
  void writeInt(int value) throws IOException;

  /** Writes a 64-bit little-endian long and advances by 8 bytes. */
  void writeLong(long value) throws IOException;

  /** Writes a 32-bit little-endian float and advances by 4 bytes. */
  void writeFloat(float value) throws IOException;

  /** Writes a 16-bit little-endian short and advances by 2 bytes. */
  void writeShort(short value) throws IOException;

  /** Writes a single byte and advances by 1 byte. */
  void writeByte(byte value) throws IOException;

  /** Writes {@code count} little-endian ints from {@code src} starting at {@code offset}. */
  void writeInts(int[] src, int offset, int count) throws IOException;

  /** Writes {@code count} little-endian floats from {@code src} starting at {@code offset}. */
  void writeFloats(float[] src, int offset, int count) throws IOException;

  /** Writes {@code count} bytes from {@code src} starting at {@code offset}. */
  void writeBytes(byte[] src, int offset, int count) throws IOException;

  /**
   * Writes {@code count} floats from a MemorySegment at the given byte offset. For off-heap sources
   * this can avoid unnecessary copies.
   */
  void writeFloats(MemorySegment src, long srcByteOffset, int count) throws IOException;

  /** Writes {@code count} zero bytes at the current position (for padding/alignment). */
  void writeZeros(int count) throws IOException;

  /** Flushes any buffered data to the underlying storage. */
  void flush() throws IOException;

  @Override
  void close() throws IOException;
}
