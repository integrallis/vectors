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
package com.integrallis.vectors.storage.store;

import com.integrallis.vectors.core.VectorEncoding;
import com.integrallis.vectors.storage.StorageLayouts;
import com.integrallis.vectors.storage.io.MappedSegmentInputSupplier;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;

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
    try {
      MemorySegment seg = supplier.segment();
      int rawBytes = vectorByteSize(size, dimension, encoding);
      long stride = AlignmentUtil.alignUp(rawBytes, alignment);
      validateMappedSize(size, rawBytes, stride, dataOffset, seg.byteSize());
      return new MappedVectorStore(supplier, seg, size, dimension, encoding, stride, dataOffset);
    } catch (RuntimeException e) {
      supplier.close();
      throw e;
    }
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
    int rawBytes = vectorByteSize(size, dimension, encoding);
    long stride = AlignmentUtil.alignUp(rawBytes, alignment);
    validateMappedSize(size, rawBytes, stride, dataOffset, segment.byteSize());
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
      checkFloatDestination(dst);
      MemorySegment.copy(segment, StorageLayouts.FLOAT_LE, offset, dst, 0, dimension);
    } else if (encoding == VectorEncoding.INT8) {
      checkFloatDestination(dst);
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
    checkByteDestination(dst);
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
    checkSegmentDestination(dst, dstByteOffset);
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

  private void checkFloatDestination(float[] dst) {
    if (dst.length < dimension) {
      throw new IllegalArgumentException(
          "destination float length " + dst.length + " < required " + dimension);
    }
  }

  private void checkByteDestination(byte[] dst) {
    if (dst.length < rawVectorByteSize) {
      throw new IllegalArgumentException(
          "destination byte length " + dst.length + " < required " + rawVectorByteSize);
    }
  }

  private void checkSegmentDestination(MemorySegment dst, long dstByteOffset) {
    Objects.requireNonNull(dst, "dst");
    if (dstByteOffset < 0 || dst.byteSize() - dstByteOffset < rawVectorByteSize) {
      throw new IndexOutOfBoundsException(
          "destination range ["
              + dstByteOffset
              + ", "
              + (dstByteOffset + rawVectorByteSize)
              + ") outside segment length "
              + dst.byteSize());
    }
  }

  private static int vectorByteSize(int size, int dimension, VectorEncoding encoding) {
    if (size < 0) {
      throw new IllegalArgumentException("size must be non-negative: " + size);
    }
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive: " + dimension);
    }
    return Objects.requireNonNull(encoding, "encoding").vectorByteSize(dimension);
  }

  private static void validateMappedSize(
      int size, int rawVectorByteSize, long stride, long dataOffset, long fileSize) {
    if (dataOffset < 0) {
      throw new IllegalArgumentException("dataOffset must be non-negative: " + dataOffset);
    }
    long requiredSize;
    try {
      requiredSize =
          size == 0
              ? dataOffset
              : Math.addExact(
                  dataOffset,
                  Math.addExact(Math.multiplyExact((long) size - 1, stride), rawVectorByteSize));
    } catch (ArithmeticException e) {
      throw new IllegalArgumentException("mapped vector store byte size overflows long", e);
    }
    if (fileSize < requiredSize) {
      EOFException cause =
          new EOFException(
              "mapped vector store requires " + requiredSize + " bytes but file has " + fileSize);
      throw new IllegalArgumentException(
          "mapped vector store requires " + requiredSize + " bytes but file has " + fileSize,
          cause);
    }
  }

  @Override
  public void close() {
    supplier.close();
  }
}
