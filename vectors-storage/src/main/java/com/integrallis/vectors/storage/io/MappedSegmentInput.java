package com.integrallis.vectors.storage.io;

import com.integrallis.vectors.storage.StorageLayouts;
import java.io.IOException;
import java.lang.foreign.MemorySegment;

/**
 * {@link RandomAccessInput} backed by a {@link MemorySegment}. Provides positional reads over a
 * memory-mapped file region. Not thread-safe — each thread should obtain its own instance via
 * {@link MappedSegmentInputSupplier#open()}.
 *
 * <p>For mmap'd segments, all reads are zero-copy (no data is copied from kernel to user space
 * beyond the initial page fault). Absolute reads ({@code readIntAt}, etc.) do not change the
 * current position.
 */
public final class MappedSegmentInput implements RandomAccessInput {

  private final MemorySegment segment;
  private long pos;

  MappedSegmentInput(MemorySegment segment) {
    this.segment = segment;
    this.pos = 0;
  }

  @Override
  public void seek(long offset) throws IOException {
    if (offset < 0 || offset > segment.byteSize()) {
      throw new IOException(
          "Seek offset out of bounds: " + offset + " (size=" + segment.byteSize() + ")");
    }
    this.pos = offset;
  }

  @Override
  public long position() {
    return pos;
  }

  @Override
  public long length() {
    return segment.byteSize();
  }

  @Override
  public int readInt() {
    int v = segment.get(StorageLayouts.INT_LE, pos);
    pos += Integer.BYTES;
    return v;
  }

  @Override
  public long readLong() {
    long v = segment.get(StorageLayouts.LONG_LE, pos);
    pos += Long.BYTES;
    return v;
  }

  @Override
  public float readFloat() {
    float v = segment.get(StorageLayouts.FLOAT_LE, pos);
    pos += Float.BYTES;
    return v;
  }

  @Override
  public short readShort() {
    short v = segment.get(StorageLayouts.SHORT_LE, pos);
    pos += Short.BYTES;
    return v;
  }

  @Override
  public byte readByte() {
    byte v = segment.get(StorageLayouts.BYTE, pos);
    pos += Byte.BYTES;
    return v;
  }

  @Override
  public void readInts(int[] dst, int offset, int count) {
    for (int i = 0; i < count; i++) {
      dst[offset + i] = segment.get(StorageLayouts.INT_LE, pos);
      pos += Integer.BYTES;
    }
  }

  @Override
  public void readFloats(float[] dst, int offset, int count) {
    for (int i = 0; i < count; i++) {
      dst[offset + i] = segment.get(StorageLayouts.FLOAT_LE, pos);
      pos += Float.BYTES;
    }
  }

  @Override
  public void readBytes(byte[] dst, int offset, int count) {
    MemorySegment.copy(segment, pos, MemorySegment.ofArray(dst), (long) offset, count);
    pos += count;
  }

  @Override
  public void readFloats(MemorySegment dst, long dstByteOffset, int count) {
    long byteCount = (long) count * Float.BYTES;
    MemorySegment.copy(segment, pos, dst, dstByteOffset, byteCount);
    pos += byteCount;
  }

  @Override
  public int readIntAt(long offset) {
    return segment.get(StorageLayouts.INT_LE, offset);
  }

  @Override
  public long readLongAt(long offset) {
    return segment.get(StorageLayouts.LONG_LE, offset);
  }

  @Override
  public float readFloatAt(long offset) {
    return segment.get(StorageLayouts.FLOAT_LE, offset);
  }

  /** Returns the underlying MemorySegment for direct slice access. */
  public MemorySegment segment() {
    return segment;
  }

  @Override
  public void close() {
    // The segment is owned by the InputSupplier's Arena; nothing to close here.
  }
}
