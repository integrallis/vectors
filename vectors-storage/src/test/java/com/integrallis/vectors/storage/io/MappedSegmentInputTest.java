package com.integrallis.vectors.storage.io;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Tests for {@link MappedSegmentInput} and {@link MappedSegmentInputSupplier}. */
class MappedSegmentInputTest {

  @TempDir Path tempDir;

  @Test
  void readScalars_roundTrip() throws IOException {
    Path file = tempDir.resolve("scalars.bin");
    // Write known values in little-endian
    ByteBuffer buf = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(42);
    buf.putLong(Long.MAX_VALUE);
    buf.putFloat(3.14f);
    buf.putShort((short) 256);
    buf.put((byte) 0x7F);
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      ch.write(buf);
    }

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      try (var reader = supplier.open()) {
        assertThat(reader.position()).isZero();
        assertThat(reader.length()).isEqualTo(buf.limit());

        assertThat(reader.readInt()).isEqualTo(42);
        assertThat(reader.readLong()).isEqualTo(Long.MAX_VALUE);
        assertThat(reader.readFloat()).isEqualTo(3.14f);
        assertThat(reader.readShort()).isEqualTo((short) 256);
        assertThat(reader.readByte()).isEqualTo((byte) 0x7F);
      }
    }
  }

  @Test
  void seek_and_absoluteReads() throws IOException {
    Path file = tempDir.resolve("seek.bin");
    ByteBuffer buf = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(100); // offset 0
    buf.putInt(200); // offset 4
    buf.putFloat(1.5f); // offset 8
    buf.putLong(999L); // offset 12
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      ch.write(buf);
    }

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      try (var reader = supplier.open()) {
        // Absolute reads don't change position
        assertThat(reader.readIntAt(0)).isEqualTo(100);
        assertThat(reader.readIntAt(4)).isEqualTo(200);
        assertThat(reader.readFloatAt(8)).isEqualTo(1.5f);
        assertThat(reader.readLongAt(12)).isEqualTo(999L);
        assertThat(reader.position()).isZero();

        // Seek to middle
        reader.seek(4);
        assertThat(reader.position()).isEqualTo(4);
        assertThat(reader.readInt()).isEqualTo(200);
        assertThat(reader.position()).isEqualTo(8);
      }
    }
  }

  @Test
  void readArrays_floatsAndInts() throws IOException {
    Path file = tempDir.resolve("arrays.bin");
    int[] ints = {10, 20, 30, 40, 50};
    float[] floats = {1.1f, 2.2f, 3.3f};

    ByteBuffer buf =
        ByteBuffer.allocate(ints.length * 4 + floats.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (int v : ints) buf.putInt(v);
    for (float v : floats) buf.putFloat(v);
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      ch.write(buf);
    }

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      try (var reader = supplier.open()) {
        int[] readInts = new int[5];
        reader.readInts(readInts, 0, 5);
        assertThat(readInts).containsExactly(10, 20, 30, 40, 50);

        float[] readFloats = new float[3];
        reader.readFloats(readFloats, 0, 3);
        assertThat(readFloats).containsExactly(1.1f, 2.2f, 3.3f);
      }
    }
  }

  @Test
  void readBytes_bulk() throws IOException {
    Path file = tempDir.resolve("bytes.bin");
    byte[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Files.write(file, data);

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      try (var reader = supplier.open()) {
        byte[] readBack = new byte[10];
        reader.readBytes(readBack, 0, 10);
        assertThat(readBack).containsExactly(data);
      }
    }
  }

  @Test
  void readFloats_intoMemorySegment() throws IOException {
    Path file = tempDir.resolve("memseg.bin");
    float[] floats = {1.0f, 2.0f, 3.0f, 4.0f};
    ByteBuffer buf = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : floats) buf.putFloat(v);
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      ch.write(buf);
    }

    try (var supplier = MappedSegmentInputSupplier.open(file);
        Arena arena = Arena.ofConfined()) {
      try (var reader = supplier.open()) {
        MemorySegment dst = arena.allocate(16);
        reader.readFloats(dst, 0, 4);
        // Verify the segment has the float data
        for (int i = 0; i < 4; i++) {
          assertThat(dst.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i))
              .isEqualTo(floats[i]);
        }
      }
    }
  }

  @Test
  void seek_beyondLength_throwsIOException() throws IOException {
    Path file = tempDir.resolve("small.bin");
    Files.write(file, new byte[] {1, 2, 3, 4});

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      try (var reader = supplier.open()) {
        assertThatThrownBy(() -> reader.seek(100)).isInstanceOf(IOException.class);
      }
    }
  }

  @Test
  void multipleReaders_independent() throws IOException {
    Path file = tempDir.resolve("multi.bin");
    ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    buf.putInt(111);
    buf.putInt(222);
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      ch.write(buf);
    }

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      try (var r1 = supplier.open();
          var r2 = supplier.open()) {
        // r1 reads first int
        assertThat(r1.readInt()).isEqualTo(111);
        assertThat(r1.position()).isEqualTo(4);

        // r2 is independent, starts at 0
        assertThat(r2.readInt()).isEqualTo(111);
        assertThat(r2.position()).isEqualTo(4);

        // r1 reads second int
        assertThat(r1.readInt()).isEqualTo(222);
        // r2 reads second int independently
        assertThat(r2.readInt()).isEqualTo(222);
      }
    }
  }

  @Test
  void supplierLength_matchesFileSize() throws IOException {
    Path file = tempDir.resolve("length.bin");
    Files.write(file, new byte[1024]);

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      assertThat(supplier.length()).isEqualTo(1024);
    }
  }

  @Test
  void concurrentReads_fromDifferentThreads() throws Exception {
    Path file = tempDir.resolve("concurrent.bin");
    int numInts = 10000;
    ByteBuffer buf = ByteBuffer.allocate(numInts * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < numInts; i++) {
      buf.putInt(i);
    }
    buf.flip();
    try (FileChannel ch =
        FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      ch.write(buf);
    }

    try (var supplier = MappedSegmentInputSupplier.open(file)) {
      int numThreads = 8;
      Thread[] threads = new Thread[numThreads];
      boolean[] results = new boolean[numThreads];

      for (int t = 0; t < numThreads; t++) {
        final int threadId = t;
        threads[t] =
            Thread.ofVirtual()
                .start(
                    () -> {
                      try (var reader = supplier.open()) {
                        // Each thread reads a different section
                        int start = threadId * (numInts / numThreads);
                        int end = start + (numInts / numThreads);
                        reader.seek((long) start * 4);
                        boolean ok = true;
                        for (int i = start; i < end; i++) {
                          if (reader.readInt() != i) {
                            ok = false;
                            break;
                          }
                        }
                        results[threadId] = ok;
                      } catch (IOException e) {
                        results[threadId] = false;
                      }
                    });
      }

      for (Thread thread : threads) {
        thread.join();
      }

      for (int t = 0; t < numThreads; t++) {
        assertThat(results[t]).as("Thread %d", t).isTrue();
      }
    }
  }
}
