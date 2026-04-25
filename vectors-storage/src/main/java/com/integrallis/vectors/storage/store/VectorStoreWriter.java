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
import com.integrallis.vectors.storage.io.ChannelOutput;
import com.integrallis.vectors.storage.memory.AlignmentUtil;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Writes contiguous vector data to a file with alignment padding. The resulting file can be opened
 * as a {@link MappedVectorStore}.
 *
 * <p>Each vector is padded to the specified alignment boundary (default 64 bytes). This ensures
 * SIMD-friendly access patterns when reading vectors from the mmap'd file.
 */
public final class VectorStoreWriter implements AutoCloseable {

  private final ChannelOutput output;
  private final VectorEncoding encoding;
  private final int dimension;
  private final int rawVectorByteSize;
  private final int paddingSize;
  private int count;

  private VectorStoreWriter(
      ChannelOutput output, VectorEncoding encoding, int dimension, int alignment) {
    this.output = output;
    this.encoding = encoding;
    this.dimension = dimension;
    this.rawVectorByteSize = encoding.vectorByteSize(dimension);
    long stride = AlignmentUtil.alignUp(rawVectorByteSize, alignment);
    this.paddingSize = (int) (stride - rawVectorByteSize);
    this.count = 0;
  }

  /**
   * Opens a writer with default 64-byte alignment.
   *
   * @param path the output file
   * @param dimension the number of dimensions per vector
   * @param encoding the vector encoding
   * @return a new writer
   * @throws IOException if the file cannot be opened
   */
  public static VectorStoreWriter open(Path path, int dimension, VectorEncoding encoding)
      throws IOException {
    return open(path, dimension, encoding, AlignmentUtil.VECTOR_ALIGNMENT);
  }

  /**
   * Opens a writer with explicit alignment.
   *
   * @param path the output file
   * @param dimension the number of dimensions per vector
   * @param encoding the vector encoding
   * @param alignment per-vector alignment (must be a power of two)
   * @return a new writer
   * @throws IOException if the file cannot be opened
   */
  public static VectorStoreWriter open(
      Path path, int dimension, VectorEncoding encoding, int alignment) throws IOException {
    ChannelOutput output = ChannelOutput.open(path);
    return new VectorStoreWriter(output, encoding, dimension, alignment);
  }

  /**
   * Appends a float vector. Only valid for {@link VectorEncoding#FLOAT32}.
   *
   * @param vector the float vector to write
   * @throws IOException if writing fails
   * @throws IllegalArgumentException if dimension doesn't match or encoding is wrong
   */
  public void writeVector(float[] vector) throws IOException {
    if (encoding != VectorEncoding.FLOAT32) {
      throw new IllegalArgumentException("writeVector(float[]) requires FLOAT32 encoding");
    }
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Vector dimension " + vector.length + " != expected " + dimension);
    }
    output.writeFloats(vector, 0, dimension);
    if (paddingSize > 0) {
      output.writeZeros(paddingSize);
    }
    count++;
  }

  /**
   * Appends a byte vector. Valid for {@link VectorEncoding#INT8} or raw bytes.
   *
   * @param vector the byte vector to write
   * @throws IOException if writing fails
   */
  public void writeVector(byte[] vector) throws IOException {
    if (vector.length != rawVectorByteSize) {
      throw new IllegalArgumentException(
          "Vector byte size " + vector.length + " != expected " + rawVectorByteSize);
    }
    output.writeBytes(vector, 0, vector.length);
    if (paddingSize > 0) {
      output.writeZeros(paddingSize);
    }
    count++;
  }

  /** Returns the number of vectors written so far. */
  public int count() {
    return count;
  }

  @Override
  public void close() throws IOException {
    output.flush();
    output.close();
  }
}
