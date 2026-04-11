package com.integrallis.vectors.db.storage;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.zip.CRC32;

/**
 * Thin wrapper around {@link java.util.zip.CRC32} for file-level integrity checks.
 *
 * <p>CRC32 is chosen for zero-dependency simplicity rather than raw speed. It is not on the SIMD
 * scoring hot path — it runs exactly twice per generation lifecycle (once at commit time when
 * writing {@code manifest.bin}, once at open time when validating the same files). On modern SSDs
 * this is I/O-bound, not CPU-bound, so a faster hash like xxHash3 would not meaningfully reduce
 * end-to-end commit latency.
 */
public final class Checksums {

  private Checksums() {}

  /**
   * Result of a single-pass file hash: the number of bytes read from the file and the CRC32 of
   * those bytes. Returned together by {@link #ofFile(Path)} so callers that need to check both
   * against manifest-declared values (e.g. payload integrity verification during crash recovery) do
   * so from a single sequential read — eliminating the TOCTOU window between a standalone {@code
   * Files.size} stat call and a subsequent {@code ofFile} read.
   *
   * <p>The two fields are consistent by construction: {@code length} is the exact number of bytes
   * that contributed to {@code crc32}, not a potentially-stale stat result.
   */
  public record FileChecksum(long length, long crc32) {}

  /**
   * Computes the CRC32 of a full file and reports the number of bytes that contributed to it. Reads
   * sequentially in 64 KiB chunks. The returned {@link FileChecksum#length} is the exact byte count
   * consumed from the file channel — not a stat call — so callers can validate both length and CRC
   * against manifest-declared values without a stat/read TOCTOU gap.
   */
  public static FileChecksum ofFile(Path file) throws IOException {
    CRC32 crc = new CRC32();
    long total = 0L;
    try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
      ByteBuffer buf = ByteBuffer.allocateDirect(64 * 1024);
      int read;
      while ((read = ch.read(buf)) != -1) {
        buf.flip();
        crc.update(buf);
        total += read;
        buf.clear();
      }
    }
    return new FileChecksum(total, crc.getValue());
  }

  /** Computes the CRC32 of a byte array slice. */
  public static long ofBytes(byte[] bytes, int offset, int length) {
    CRC32 crc = new CRC32();
    crc.update(bytes, offset, length);
    return crc.getValue();
  }

  /** Computes the CRC32 of the entire byte array. */
  public static long ofBytes(byte[] bytes) {
    return ofBytes(bytes, 0, bytes.length);
  }

  /**
   * Computes the CRC32 of {@code buffer} from its current position to its limit.
   *
   * <p><b>Side effect:</b> {@link CRC32#update(ByteBuffer)} advances {@code buffer}'s position to
   * its limit. Callers that need to retain the original position must {@link ByteBuffer#duplicate
   * duplicate} the buffer first, or rewind it explicitly after this call returns.
   */
  public static long ofBuffer(ByteBuffer buffer) {
    CRC32 crc = new CRC32();
    crc.update(buffer);
    return crc.getValue();
  }

  /**
   * Computes the CRC32 of a {@link MemorySegment} slice {@code [offset, offset + length)}. Copies
   * in 64 KiB chunks through an on-heap staging buffer — {@link CRC32#update(ByteBuffer)} accepts a
   * {@link ByteBuffer} but {@link CRC32#update(byte[], int, int)} is slightly faster and avoids JNI
   * overhead per chunk.
   */
  public static long ofSegment(MemorySegment segment, long offset, long length) {
    if (offset < 0 || length < 0 || offset + length > segment.byteSize()) {
      throw new IndexOutOfBoundsException(
          "CRC32 slice out of bounds: offset=" + offset + ", length=" + length);
    }
    CRC32 crc = new CRC32();
    byte[] stage = new byte[64 * 1024];
    long remaining = length;
    long pos = offset;
    while (remaining > 0) {
      int chunk = (int) Math.min(remaining, stage.length);
      MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, pos, stage, 0, chunk);
      crc.update(stage, 0, chunk);
      pos += chunk;
      remaining -= chunk;
    }
    return crc.getValue();
  }

  /** Computes the CRC32 of the entire {@link MemorySegment}. */
  public static long ofSegment(MemorySegment segment) {
    return ofSegment(segment, 0L, segment.byteSize());
  }
}
