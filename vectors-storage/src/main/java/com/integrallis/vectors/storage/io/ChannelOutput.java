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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * {@link RandomAccessOutput} backed by a {@link FileChannel}. Writes use a small reusable buffer
 * for scalar values and direct bulk writes for arrays. All multi-byte values are written in
 * little-endian byte order.
 */
public final class ChannelOutput implements RandomAccessOutput {

  private static final int SCALAR_BUFFER_SIZE = 8; // enough for a long
  private static final int ZERO_BUFFER_SIZE = 4096;

  /** Shared, preallocated, read-only buffer of zero bytes used for padding writes. */
  private static final ByteBuffer SHARED_ZEROS =
      ByteBuffer.allocateDirect(ZERO_BUFFER_SIZE).asReadOnlyBuffer();

  private final FileChannel channel;
  private final ByteBuffer scalarBuf;

  /** Reusable little-endian scratch buffer for bulk primitive writes; grows on demand. */
  private ByteBuffer bulkBuf;

  ChannelOutput(FileChannel channel) {
    this.channel = channel;
    this.scalarBuf = ByteBuffer.allocate(SCALAR_BUFFER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
  }

  /**
   * Opens a file for writing. Creates the file if it doesn't exist; truncates if it does.
   *
   * @param path the file to write to
   * @return a new output
   * @throws IOException if the file cannot be opened
   */
  public static ChannelOutput open(Path path) throws IOException {
    FileChannel ch =
        FileChannel.open(
            path,
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE,
            StandardOpenOption.TRUNCATE_EXISTING);
    return new ChannelOutput(ch);
  }

  @Override
  public void seek(long offset) throws IOException {
    channel.position(offset);
  }

  @Override
  public long position() throws IOException {
    return channel.position();
  }

  @Override
  public void writeInt(int value) throws IOException {
    scalarBuf.clear().putInt(value).flip();
    writeFully(scalarBuf);
  }

  @Override
  public void writeLong(long value) throws IOException {
    scalarBuf.clear().putLong(value).flip();
    writeFully(scalarBuf);
  }

  @Override
  public void writeFloat(float value) throws IOException {
    scalarBuf.clear().putFloat(value).flip();
    writeFully(scalarBuf);
  }

  @Override
  public void writeShort(short value) throws IOException {
    scalarBuf.clear().putShort(value).flip();
    writeFully(scalarBuf);
  }

  @Override
  public void writeByte(byte value) throws IOException {
    scalarBuf.clear().put(value).flip();
    writeFully(scalarBuf);
  }

  @Override
  public void writeInts(int[] src, int offset, int count) throws IOException {
    int bytes = count * Integer.BYTES;
    ByteBuffer buf = bulkBuffer(bytes);
    buf.asIntBuffer().put(src, offset, count); // bulk copy, no per-element loop
    buf.limit(bytes);
    writeFully(buf);
  }

  @Override
  public void writeFloats(float[] src, int offset, int count) throws IOException {
    int bytes = count * Float.BYTES;
    ByteBuffer buf = bulkBuffer(bytes);
    buf.asFloatBuffer().put(src, offset, count); // bulk, native-order copy, no per-element loop
    buf.limit(bytes);
    writeFully(buf);
  }

  /**
   * Writes the remaining bytes of {@code buf} (position..limit) to the channel, advancing the
   * buffer to its limit. The buffer's byte order must already match the on-disk layout. Lets
   * callers that own a prebuilt buffer (e.g. a padded vector stride) write it in a single call.
   */
  public void write(ByteBuffer buf) throws IOException {
    writeFully(buf);
  }

  @Override
  public void writeBytes(byte[] src, int offset, int count) throws IOException {
    ByteBuffer buf = ByteBuffer.wrap(src, offset, count);
    writeFully(buf);
  }

  @Override
  public void writeFloats(MemorySegment src, long srcByteOffset, int count) throws IOException {
    long byteCount = (long) count * Float.BYTES;
    writeFully(src.asSlice(srcByteOffset, byteCount).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN));
  }

  @Override
  public void writeZeros(int count) throws IOException {
    if (count <= 0) return;
    // Write from the shared preallocated zero buffer; duplicate() gives an independent
    // position/limit view without copying or reallocating the zero payload.
    int remaining = count;
    while (remaining > 0) {
      int toWrite = Math.min(remaining, ZERO_BUFFER_SIZE);
      ByteBuffer zeros = SHARED_ZEROS.duplicate();
      zeros.position(0).limit(toWrite);
      writeFully(zeros);
      remaining -= toWrite;
    }
  }

  @Override
  public void flush() throws IOException {
    channel.force(true);
  }

  @Override
  public void close() throws IOException {
    channel.close();
  }

  private void writeFully(ByteBuffer buf) throws IOException {
    while (buf.hasRemaining()) {
      channel.write(buf);
    }
  }

  /**
   * Returns a reusable little-endian scratch buffer with at least {@code minBytes} capacity,
   * cleared (position 0, limit = capacity). Grows the backing buffer only when a larger one is
   * needed.
   */
  private ByteBuffer bulkBuffer(int minBytes) {
    ByteBuffer b = bulkBuf;
    if (b == null || b.capacity() < minBytes) {
      b = ByteBuffer.allocate(minBytes).order(ByteOrder.LITTLE_ENDIAN);
      bulkBuf = b;
    }
    b.clear();
    return b;
  }
}
