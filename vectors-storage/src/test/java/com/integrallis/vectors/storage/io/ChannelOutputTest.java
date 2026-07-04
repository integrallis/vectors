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

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Tests for {@link ChannelOutput}. */
class ChannelOutputTest {

  @TempDir Path tempDir;

  @Test
  void writeScalars_readBack() throws IOException {
    Path file = tempDir.resolve("scalars.bin");
    try (var out = ChannelOutput.open(file)) {
      out.writeInt(42);
      out.writeLong(Long.MAX_VALUE);
      out.writeFloat(3.14f);
      out.writeShort((short) 256);
      out.writeByte((byte) 0x7F);
    }

    // Read back and verify little-endian
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();

      assertThat(buf.getInt()).isEqualTo(42);
      assertThat(buf.getLong()).isEqualTo(Long.MAX_VALUE);
      assertThat(buf.getFloat()).isEqualTo(3.14f);
      assertThat(buf.getShort()).isEqualTo((short) 256);
      assertThat(buf.get()).isEqualTo((byte) 0x7F);
    }
  }

  @Test
  void writeArrays_readBack() throws IOException {
    Path file = tempDir.resolve("arrays.bin");
    float[] floats = {1.1f, 2.2f, 3.3f, 4.4f};
    int[] ints = {10, 20, 30};
    byte[] bytes = {1, 2, 3, 4, 5};

    try (var out = ChannelOutput.open(file)) {
      out.writeFloats(floats, 0, 4);
      out.writeInts(ints, 0, 3);
      out.writeBytes(bytes, 0, 5);
    }

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();

      for (float f : floats) assertThat(buf.getFloat()).isEqualTo(f);
      for (int i : ints) assertThat(buf.getInt()).isEqualTo(i);
      for (byte b : bytes) assertThat(buf.get()).isEqualTo(b);
    }
  }

  @Test
  void writeZeros_writesCorrectPadding() throws IOException {
    Path file = tempDir.resolve("zeros.bin");
    try (var out = ChannelOutput.open(file)) {
      out.writeInt(0xDEADBEEF);
      out.writeZeros(12);
      out.writeInt(0xCAFEBABE);
    }

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      assertThat(ch.size()).isEqualTo(4 + 12 + 4);
      ByteBuffer buf = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      assertThat(buf.getInt()).isEqualTo(0xDEADBEEF);
      for (int i = 0; i < 12; i++) {
        assertThat(buf.get()).as("padding byte %d", i).isZero();
      }
      assertThat(buf.getInt()).isEqualTo(0xCAFEBABE);
    }
  }

  @Test
  void seekAndPosition() throws IOException {
    Path file = tempDir.resolve("seek.bin");
    try (var out = ChannelOutput.open(file)) {
      out.writeInt(100);
      assertThat(out.position()).isEqualTo(4);
      out.writeInt(200);
      assertThat(out.position()).isEqualTo(8);

      // Seek back and overwrite
      out.seek(0);
      assertThat(out.position()).isZero();
      out.writeInt(999);
    }

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      assertThat(buf.getInt()).isEqualTo(999); // overwritten
      assertThat(buf.getInt()).isEqualTo(200); // unchanged
    }
  }

  @Test
  void writeFloats_fromMemorySegment() throws IOException {
    Path file = tempDir.resolve("memseg.bin");
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(16);
      for (int i = 0; i < 4; i++) {
        seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (i + 1));
      }

      try (var out = ChannelOutput.open(file)) {
        out.writeFloats(seg, 0, 4);
      }
    }

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      for (int i = 0; i < 4; i++) {
        assertThat(buf.getFloat()).isEqualTo((float) (i + 1));
      }
    }
  }

  @Test
  void writeFloats_fromMemorySegmentHonorsByteOffset() throws IOException {
    Path file = tempDir.resolve("memseg_offset.bin");
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(24);
      for (int i = 0; i < 6; i++) {
        seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) (i + 1));
      }

      try (var out = ChannelOutput.open(file)) {
        out.writeFloats(seg, Float.BYTES * 2L, 3);
      }
    }

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      assertThat(buf.getFloat()).isEqualTo(3.0f);
      assertThat(buf.getFloat()).isEqualTo(4.0f);
      assertThat(buf.getFloat()).isEqualTo(5.0f);
    }
  }

  @Test
  void roundTrip_writeAndReadViaMappedInput() throws IOException {
    Path file = tempDir.resolve("roundtrip.bin");
    try (var out = ChannelOutput.open(file)) {
      out.writeInt(123);
      out.writeFloat(4.56f);
      out.writeLong(789L);
    }

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      try (var reader = supplier.open()) {
        assertThat(reader.readInt()).isEqualTo(123);
        assertThat(reader.readFloat()).isEqualTo(4.56f);
        assertThat(reader.readLong()).isEqualTo(789L);
      }
    }
  }

  @Test
  void flushForcesFileDataAndMetadata() throws IOException {
    RecordingFileChannel channel = new RecordingFileChannel();

    try (var out = new ChannelOutput(channel)) {
      out.flush();
    }

    assertThat(channel.forceCalled).isTrue();
    assertThat(channel.forceMetadata).isTrue();
  }

  @Test
  void writeSubArrays_offsetAndCount() throws IOException {
    Path file = tempDir.resolve("subarrays.bin");
    float[] floats = {0.0f, 1.1f, 2.2f, 3.3f, 4.4f};
    int[] ints = {0, 10, 20, 30, 40};

    try (var out = ChannelOutput.open(file)) {
      // Write only elements [1..3]
      out.writeFloats(floats, 1, 3);
      out.writeInts(ints, 2, 2);
    }

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      assertThat(buf.getFloat()).isEqualTo(1.1f);
      assertThat(buf.getFloat()).isEqualTo(2.2f);
      assertThat(buf.getFloat()).isEqualTo(3.3f);
      assertThat(buf.getInt()).isEqualTo(20);
      assertThat(buf.getInt()).isEqualTo(30);
    }
  }

  @Test
  void writePrebuiltByteBuffer_readBack() throws IOException {
    Path file = tempDir.resolve("prebuilt.bin");
    ByteBuffer body = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
    body.putFloat(1.5f).putFloat(2.5f).putInt(7);
    body.flip();
    try (var out = ChannelOutput.open(file)) {
      out.write(body);
    }
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      assertThat(buf.getFloat()).isEqualTo(1.5f);
      assertThat(buf.getFloat()).isEqualTo(2.5f);
      assertThat(buf.getInt()).isEqualTo(7);
    }
  }

  @Test
  void writeFloatsAndInts_growReusableBulkBuffer() throws IOException {
    Path file = tempDir.resolve("grow.bin");
    float[] small = {1f, 2f};
    float[] large = new float[2048]; // exceeds the initial bulk buffer -> forces a grow
    for (int i = 0; i < large.length; i++) large[i] = i * 0.5f;
    int[] ints = new int[1024];
    for (int i = 0; i < ints.length; i++) ints[i] = i;

    try (var out = ChannelOutput.open(file)) {
      out.writeFloats(small, 0, small.length); // allocates a small bulk buffer
      out.writeFloats(large, 0, large.length); // regrows the bulk buffer
      out.writeInts(ints, 0, ints.length); // reuses the larger buffer
    }

    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      for (float f : small) assertThat(buf.getFloat()).isEqualTo(f);
      for (float f : large) assertThat(buf.getFloat()).isEqualTo(f);
      for (int i : ints) assertThat(buf.getInt()).isEqualTo(i);
    }
  }

  @Test
  void writeZeros_largePaddingSpansSharedZeroBuffer() throws IOException {
    Path file = tempDir.resolve("bigzeros.bin");
    int pad = 10_000; // > the 4096-byte shared zero buffer -> multiple chunked writes
    try (var out = ChannelOutput.open(file)) {
      out.writeInt(1);
      out.writeZeros(pad);
      out.writeInt(2);
    }
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      assertThat(ch.size()).isEqualTo(4L + pad + 4);
      ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(ByteOrder.LITTLE_ENDIAN);
      ch.read(buf);
      buf.flip();
      assertThat(buf.getInt()).isEqualTo(1);
      for (int i = 0; i < pad; i++) assertThat(buf.get()).as("padding byte %d", i).isZero();
      assertThat(buf.getInt()).isEqualTo(2);
    }
  }

  private static final class RecordingFileChannel extends FileChannel {
    private boolean forceCalled;
    private boolean forceMetadata;
    private long position;

    @Override
    public int read(ByteBuffer dst) {
      return -1;
    }

    @Override
    public long read(ByteBuffer[] dsts, int offset, int length) {
      return -1;
    }

    @Override
    public int write(ByteBuffer src) {
      int remaining = src.remaining();
      src.position(src.limit());
      position += remaining;
      return remaining;
    }

    @Override
    public long write(ByteBuffer[] srcs, int offset, int length) {
      long written = 0;
      for (int i = 0; i < length; i++) {
        written += write(srcs[offset + i]);
      }
      return written;
    }

    @Override
    public long position() {
      return position;
    }

    @Override
    public FileChannel position(long newPosition) {
      position = newPosition;
      return this;
    }

    @Override
    public long size() {
      return position;
    }

    @Override
    public FileChannel truncate(long size) {
      position = Math.min(position, size);
      return this;
    }

    @Override
    public void force(boolean metaData) {
      forceCalled = true;
      forceMetadata = metaData;
    }

    @Override
    public long transferTo(long position, long count, WritableByteChannel target) {
      return 0;
    }

    @Override
    public long transferFrom(ReadableByteChannel src, long position, long count) {
      return 0;
    }

    @Override
    public int read(ByteBuffer dst, long position) {
      return -1;
    }

    @Override
    public int write(ByteBuffer src, long position) {
      return write(src);
    }

    @Override
    public MappedByteBuffer map(MapMode mode, long position, long size) {
      throw new UnsupportedOperationException();
    }

    @Override
    public FileLock lock(long position, long size, boolean shared) {
      throw new UnsupportedOperationException();
    }

    @Override
    public FileLock tryLock(long position, long size, boolean shared) {
      throw new UnsupportedOperationException();
    }

    @Override
    protected void implCloseChannel() {}
  }
}
