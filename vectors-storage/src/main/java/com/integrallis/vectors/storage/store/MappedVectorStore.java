package com.integrallis.vectors.storage.store;

import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.storage.StorageLayouts;
import com.integrallis.vectors.storage.io.MappedSegmentInputSupplier;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/**
 * Read-only {@link VectorStore} backed by a memory-mapped file. Vectors are stored contiguously
 * with optional alignment padding. All reads are zero-copy (served directly from the mmap'd
 * region).
 *
 * <p>Thread-safe: multiple threads can read concurrently. The underlying mmap segment is shared via
 * {@link java.lang.foreign.Arena#ofShared()}.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * try (var store = MappedVectorStore.open(path, 1000, 128, VectorEncoding.FLOAT32)) {
 *     float[] vec = new float[128];
 *     store.getVector(42, vec);
 * }
 * }</pre>
 */
public final class MappedVectorStore implements VectorStore {

  private final MappedSegmentInputSupplier supplier;
  private final MemorySegment segment;
  private final int size;
  private final int dimension;
  private final VectorEncoding encoding;
  private final int rawVectorByteSize;
  private final long stride;
  private final long dataOffset;

  private MappedVectorStore(
      MappedSegmentInputSupplier supplier,
      MemorySegment segment,
      int size,
      int dimension,
      VectorEncoding encoding,
      long stride,
      long dataOffset) {
    this.supplier = supplier;
    this.segment = segment;
    this.size = size;
    this.dimension = dimension;
    this.encoding = encoding;
    this.rawVectorByteSize = encoding.vectorByteSize(dimension);
    this.stride = stride;
    this.dataOffset = dataOffset;
  }

  /**
   * Opens a memory-mapped vector store from a raw vector data file. Vectors are stored starting at
   * byte offset 0 with 64-byte alignment per vector.
   *
   * @param path the file containing contiguous vector data
   * @param size the number of vectors
   * @param dimension the number of dimensions per vector
   * @param encoding the vector encoding
   * @return a new mapped vector store
   * @throws IOException if the file cannot be opened or mapped
   */
  public static MappedVectorStore open(Path path, int size, int dimension, VectorEncoding encoding)
      throws IOException {
    return open(path, size, dimension, encoding, 0, AlignmentUtil.VECTOR_ALIGNMENT);
  }

  /**
   * Opens a memory-mapped vector store with explicit data offset and alignment.
   *
   * @param path the file containing vector data
   * @param size the number of vectors
   * @param dimension the number of dimensions per vector
   * @param encoding the vector encoding
   * @param dataOffset byte offset where vector data begins in the file
   * @param alignment per-vector alignment (must be a power of two)
   * @return a new mapped vector store
   * @throws IOException if the file cannot be opened or mapped
   */
  public static MappedVectorStore open(
      Path path, int size, int dimension, VectorEncoding encoding, long dataOffset, int alignment)
      throws IOException {
    MappedSegmentInputSupplier supplier = MappedSegmentInputSupplier.open(path);
    MemorySegment seg = supplier.segment();
    int rawBytes = encoding.vectorByteSize(dimension);
    long stride = AlignmentUtil.alignUp(rawBytes, alignment);
    return new MappedVectorStore(supplier, seg, size, dimension, encoding, stride, dataOffset);
  }

  /**
   * Creates a mapped vector store from an existing supplier and segment. Useful when the vector
   * data is a section within a larger file.
   *
   * @param supplier the input supplier (takes ownership)
   * @param segment the segment containing vector data
   * @param size the number of vectors
   * @param dimension the number of dimensions per vector
   * @param encoding the vector encoding
   * @param dataOffset byte offset within the segment where vectors begin
   * @param alignment per-vector alignment
   * @return a new mapped vector store
   */
  public static MappedVectorStore wrap(
      MappedSegmentInputSupplier supplier,
      MemorySegment segment,
      int size,
      int dimension,
      VectorEncoding encoding,
      long dataOffset,
      int alignment) {
    int rawBytes = encoding.vectorByteSize(dimension);
    long stride = AlignmentUtil.alignUp(rawBytes, alignment);
    return new MappedVectorStore(supplier, segment, size, dimension, encoding, stride, dataOffset);
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public VectorEncoding encoding() {
    return encoding;
  }

  @Override
  public int vectorByteSize() {
    return rawVectorByteSize;
  }

  @Override
  public void getVector(int ordinal, float[] dst) {
    checkOrdinal(ordinal);
    long offset = vectorOffset(ordinal);
    if (encoding == VectorEncoding.FLOAT32) {
      for (int i = 0; i < dimension; i++) {
        dst[i] = segment.get(StorageLayouts.FLOAT_LE, offset + (long) i * Float.BYTES);
      }
    } else if (encoding == VectorEncoding.INT8) {
      for (int i = 0; i < dimension; i++) {
        dst[i] = segment.get(StorageLayouts.BYTE, offset + i);
      }
    } else {
      throw new UnsupportedOperationException("getVector(float[]) not supported for " + encoding);
    }
  }

  @Override
  public void getVector(int ordinal, byte[] dst) {
    checkOrdinal(ordinal);
    long offset = vectorOffset(ordinal);
    MemorySegment.copy(segment, offset, MemorySegment.ofArray(dst), 0, rawVectorByteSize);
  }

  @Override
  public MemorySegment vectorSlice(int ordinal) {
    checkOrdinal(ordinal);
    return segment.asSlice(vectorOffset(ordinal), rawVectorByteSize);
  }

  @Override
  public void getVector(int ordinal, MemorySegment dst, long dstByteOffset) {
    checkOrdinal(ordinal);
    long offset = vectorOffset(ordinal);
    MemorySegment.copy(segment, offset, dst, dstByteOffset, rawVectorByteSize);
  }

  /** Returns the byte stride between consecutive vectors (includes alignment padding). */
  public long stride() {
    return stride;
  }

  private long vectorOffset(int ordinal) {
    return dataOffset + ordinal * stride;
  }

  private void checkOrdinal(int ordinal) {
    if (ordinal < 0 || ordinal >= size) {
      throw new IndexOutOfBoundsException("ordinal " + ordinal + " out of range [0, " + size + ")");
    }
  }

  @Override
  public void close() {
    supplier.close();
  }
}
