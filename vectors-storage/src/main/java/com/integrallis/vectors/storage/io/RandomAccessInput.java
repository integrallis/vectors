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
 * Random-access input for reading on-disk data structures. Stateful (has current position) and NOT
 * thread-safe. Create per-thread instances via {@link InputSupplier}.
 *
 * <p>All multi-byte reads use little-endian byte order.
 */
public interface RandomAccessInput extends AutoCloseable {

  /** Sets the current read position. */
  void seek(long offset) throws IOException;

  /** Returns the current read position. */
  long position() throws IOException;

  /** Returns the total length of the underlying data source. */
  long length() throws IOException;

  /** Reads a 32-bit little-endian int at the current position and advances by 4 bytes. */
  int readInt() throws IOException;

  /** Reads a 64-bit little-endian long at the current position and advances by 8 bytes. */
  long readLong() throws IOException;

  /** Reads a 32-bit little-endian float at the current position and advances by 4 bytes. */
  float readFloat() throws IOException;

  /** Reads a 16-bit little-endian short at the current position and advances by 2 bytes. */
  short readShort() throws IOException;

  /** Reads a single byte at the current position and advances by 1 byte. */
  byte readByte() throws IOException;

  /** Reads {@code count} little-endian ints into {@code dst} starting at {@code offset}. */
  void readInts(int[] dst, int offset, int count) throws IOException;

  /** Reads {@code count} little-endian floats into {@code dst} starting at {@code offset}. */
  void readFloats(float[] dst, int offset, int count) throws IOException;

  /** Reads {@code count} bytes into {@code dst} starting at {@code offset}. */
  void readBytes(byte[] dst, int offset, int count) throws IOException;

  /**
   * Reads {@code count} floats directly into a MemorySegment at the given byte offset. For mmap
   * implementations this can be a zero-copy slice.
   */
  void readFloats(MemorySegment dst, long dstByteOffset, int count) throws IOException;

  /**
   * Reads a 32-bit little-endian int at the given absolute offset without changing the current
   * position.
   */
  int readIntAt(long offset) throws IOException;

  /**
   * Reads a 64-bit little-endian long at the given absolute offset without changing the current
   * position.
   */
  long readLongAt(long offset) throws IOException;

  /**
   * Reads a 32-bit little-endian float at the given absolute offset without changing the current
   * position.
   */
  float readFloatAt(long offset) throws IOException;

  @Override
  void close() throws IOException;
}
