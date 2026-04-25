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
import java.lang.foreign.MemorySegment;

/**
 * Read-only vector store backed by a contiguous memory region. Supports {@link
 * VectorEncoding#FLOAT32}, {@link VectorEncoding#INT8}, and {@link VectorEncoding#BINARY}
 * encodings.
 *
 * <p>Vectors are stored contiguously and accessed by ordinal (0-based index). Implementations may
 * store vectors off-heap via mmap for zero-copy access.
 */
public interface VectorStore extends AutoCloseable {

  /** Returns the number of vectors in this store. */
  int size();

  /** Returns the number of dimensions per vector. */
  int dimension();

  /** Returns the encoding used for vectors in this store. */
  VectorEncoding encoding();

  /**
   * Returns the raw byte size of a single vector's data (without alignment padding). For FLOAT32
   * this is {@code dimension * 4}; for INT8 this is {@code dimension}; for BINARY this is {@code
   * ceil(dimension / 64) * 8}. Use {@link
   * com.integrallis.vectors.storage.store.MappedVectorStore#stride()} for the stride including
   * alignment padding.
   */
  int vectorByteSize();

  /**
   * Copies the vector at the given ordinal into the destination array.
   *
   * @param ordinal the 0-based vector index
   * @param dst the destination float array (must have at least {@link #dimension()} elements)
   * @throws IndexOutOfBoundsException if ordinal is out of range
   */
  void getVector(int ordinal, float[] dst);

  /**
   * Copies the vector at the given ordinal into the destination byte array. For INT8 encoding,
   * copies raw bytes. For FLOAT32, copies the raw float bytes.
   *
   * @param ordinal the 0-based vector index
   * @param dst the destination byte array
   * @throws IndexOutOfBoundsException if ordinal is out of range
   */
  void getVector(int ordinal, byte[] dst);

  /**
   * Returns a zero-copy slice of the underlying memory for the vector at the given ordinal. The
   * returned segment is valid for the lifetime of this store.
   *
   * @param ordinal the 0-based vector index
   * @return a memory segment pointing to the raw vector data
   * @throws IndexOutOfBoundsException if ordinal is out of range
   */
  MemorySegment vectorSlice(int ordinal);

  /**
   * Copies the vector at the given ordinal into the destination MemorySegment.
   *
   * @param ordinal the 0-based vector index
   * @param dst the destination segment
   * @param dstByteOffset the byte offset in the destination
   */
  void getVector(int ordinal, MemorySegment dst, long dstByteOffset);

  @Override
  void close();
}
