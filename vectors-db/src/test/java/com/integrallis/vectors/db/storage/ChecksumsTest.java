package com.integrallis.vectors.db.storage;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class ChecksumsTest {

  @Test
  void emptyByteArrayHasKnownCrc32() {
    // java.util.zip.CRC32 of an empty input is 0.
    assertThat(Checksums.ofBytes(new byte[0])).isEqualTo(0L);
  }

  @Test
  void bytesMatchJdkCrc32() {
    byte[] data = "hello, java-vectors".getBytes();
    CRC32 ref = new CRC32();
    ref.update(data);
    assertThat(Checksums.ofBytes(data)).isEqualTo(ref.getValue());
  }

  @Test
  void sliceMatchesJdkCrc32() {
    byte[] data = new byte[256];
    new Random(7L).nextBytes(data);
    CRC32 ref = new CRC32();
    ref.update(data, 32, 128);
    assertThat(Checksums.ofBytes(data, 32, 128)).isEqualTo(ref.getValue());
  }

  @Test
  void bufferMatchesJdkCrc32() {
    byte[] data = new byte[1024];
    new Random(13L).nextBytes(data);
    ByteBuffer buf = ByteBuffer.wrap(data);
    CRC32 ref = new CRC32();
    ref.update(data);
    assertThat(Checksums.ofBuffer(buf)).isEqualTo(ref.getValue());
  }

  @Test
  void segmentMatchesJdkCrc32() {
    Random rng = new Random(100L);
    int len = 200_000; // exercises the multi-chunk path (stage = 64 KiB)
    byte[] data = new byte[len];
    rng.nextBytes(data);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(len);
      MemorySegment.copy(data, 0, seg, ValueLayout.JAVA_BYTE, 0, len);
      CRC32 ref = new CRC32();
      ref.update(data);
      assertThat(Checksums.ofSegment(seg)).isEqualTo(ref.getValue());
    }
  }

  @Test
  void segmentSliceMatchesJdkCrc32() {
    byte[] data = new byte[128];
    new Random(21L).nextBytes(data);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(128);
      MemorySegment.copy(data, 0, seg, ValueLayout.JAVA_BYTE, 0, 128);
      CRC32 ref = new CRC32();
      ref.update(data, 16, 64);
      assertThat(Checksums.ofSegment(seg, 16L, 64L)).isEqualTo(ref.getValue());
    }
  }

  @Test
  void fileMatchesByteArray(@TempDir Path tmp) throws IOException {
    byte[] data = new byte[150_000]; // exercises multiple 64 KiB reads
    new Random(99L).nextBytes(data);
    Path file = tmp.resolve("blob.bin");
    Files.write(file, data);
    Checksums.FileChecksum actual = Checksums.ofFile(file);
    assertThat(actual.length()).isEqualTo(data.length);
    assertThat(actual.crc32()).isEqualTo(Checksums.ofBytes(data));
  }

  @Test
  void ofFileEmptyFileReportsZeroLengthAndZeroCrc(@TempDir Path tmp) throws IOException {
    // CRC32 of an empty input is 0 (per the bytes test above); the length returned by ofFile
    // must be 0 as well, and the two values must be consistent by construction (single pass).
    Path file = tmp.resolve("empty.bin");
    Files.write(file, new byte[0]);
    Checksums.FileChecksum actual = Checksums.ofFile(file);
    assertThat(actual.length()).isZero();
    assertThat(actual.crc32()).isZero();
  }

  @Test
  void ofFileReportsLengthExactlyMatchingBytesHashed(@TempDir Path tmp) throws IOException {
    // Exact boundary sizes around the 64 KiB read buffer to make sure the byte-count accumulator
    // tracks the true read size even when the last chunk is partial. 64 KiB = 65536.
    int[] sizes = {1, 65_535, 65_536, 65_537, 131_072, 200_000};
    Random rng = new Random(2026L);
    for (int size : sizes) {
      byte[] data = new byte[size];
      rng.nextBytes(data);
      Path file = tmp.resolve("blob-" + size + ".bin");
      Files.write(file, data);
      Checksums.FileChecksum actual = Checksums.ofFile(file);
      assertThat(actual.length()).as("length for size=%d", size).isEqualTo(size);
      assertThat(actual.crc32()).as("crc for size=%d", size).isEqualTo(Checksums.ofBytes(data));
    }
  }
}
