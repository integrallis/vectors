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
import java.nio.channels.FileChannel;
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
}
