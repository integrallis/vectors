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

import com.integrallis.vectors.storage.StorageLayouts;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
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

  private final FileChannel channel;
  private final ByteBuffer scalarBuf;

  private ChannelOutput(FileChannel channel) {
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

  /**
   * Opens a file for writing without truncating (for appending or random writes).
   *
   * @param path the file to write to
   * @return a new output
   * @throws IOException if the file cannot be opened
   */
  public static ChannelOutput openExisting(Path path) throws IOException {
    FileChannel ch = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
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
    ByteBuffer buf = ByteBuffer.allocate(count * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putInt(src[offset + i]);
    }
    buf.flip();
    writeFully(buf);
  }

  @Override
  public void writeFloats(float[] src, int offset, int count) throws IOException {
    ByteBuffer buf = ByteBuffer.allocate(count * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(src[offset + i]);
    }
    buf.flip();
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
    // Copy from MemorySegment to a heap array, then write
    try (Arena arena = Arena.ofConfined()) {
      SegmentAllocator alloc = SegmentAllocator.slicingAllocator(arena.allocate(byteCount));
      MemorySegment tmp = alloc.allocate(byteCount);
      MemorySegment.copy(src, srcByteOffset, tmp, 0, byteCount);
      float[] floats = new float[count];
      for (int i = 0; i < count; i++) {
        floats[i] = tmp.get(StorageLayouts.FLOAT_LE, (long) i * Float.BYTES);
      }
      writeFloats(floats, 0, count);
    }
  }

  @Override
  public void writeZeros(int count) throws IOException {
    if (count <= 0) return;
    // Write zeros in chunks to avoid large allocations
    int chunkSize = Math.min(count, 4096);
    ByteBuffer zeros = ByteBuffer.allocate(chunkSize);
    int remaining = count;
    while (remaining > 0) {
      int toWrite = Math.min(remaining, chunkSize);
      zeros.clear().limit(toWrite);
      writeFully(zeros);
      remaining -= toWrite;
    }
  }

  @Override
  public void flush() throws IOException {
    channel.force(false);
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
}
