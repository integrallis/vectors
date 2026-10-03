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
package com.integrallis.vectors.bench.dataset;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads the big-ann-benchmarks "competition format" binaries used by the NeurIPS'21 and '23 tracks.
 *
 * <p>Two layouts, both little-endian:
 *
 * <ul>
 *   <li><b>Vectors</b> ({@code *.bin}, also called {@code fbin}): {@code uint32 numPoints}, {@code
 *       uint32 numDimensions}, then {@code numPoints * numDimensions} {@code float32}.
 *   <li><b>Ground truth</b>: {@code uint32 numQueries}, {@code uint32 k}, then {@code numQueries *
 *       k} {@code uint32} neighbor ids, then {@code numQueries * k} {@code float32} distances. The
 *       distances block is skipped here; recall only needs the ids.
 * </ul>
 *
 * <p>This format is what unlocks the datasets ANN-Benchmarks does not publish as HDF5 — notably the
 * Cohere Wikipedia set at 768 dimensions, which is the only published million-scale corpus sitting
 * in the 512-768 range we actually ship against.
 *
 * <p><b>Cropped files.</b> big-ann serves one base file per corpus and derives smaller scales by
 * taking a prefix of it, so a 1M subset of a 35M file is its first 1M vectors. A prefix fetched by
 * HTTP range still carries the original header, which would then claim 35M points in a 3 GB file;
 * the fetch is expected to rewrite the count. {@link #readVectors(Path)} trusts the header and
 * verifies it against the file length, so a file whose header and size disagree fails loudly here
 * instead of silently reading past the end.
 */
public final class BigAnnLoader {

  private static final int HEADER_BYTES = 8;

  private BigAnnLoader() {}

  /** Number of points and dimensions declared by a vector file's header. */
  public record Shape(int numPoints, int dimensions) {}

  /** Reads the header of a vector file without reading its payload. */
  public static Shape readShape(Path path) {
    try (DataInputStream in =
        new DataInputStream(new BufferedInputStream(Files.newInputStream(path), HEADER_BYTES))) {
      int count = readLeInt(in);
      int dimensions = readLeInt(in);
      if (count < 0 || dimensions <= 0)
        throw new IllegalStateException("invalid vector header: " + path);
      return new Shape(count, dimensions);
    } catch (IOException e) {
      throw new UncheckedIOException("reading header of " + path, e);
    }
  }

  /**
   * Reads every vector in a competition-format vector file.
   *
   * @throws IllegalStateException if the header disagrees with the file length, which is what a
   *     truncated or un-rewritten cropped download looks like
   */
  public static float[][] readVectors(Path path) {
    Shape shape = readShape(path);
    long expected =
        (long) HEADER_BYTES + (long) shape.numPoints() * shape.dimensions() * Float.BYTES;
    long actual;
    try {
      actual = Files.size(path);
    } catch (IOException e) {
      throw new UncheckedIOException("sizing " + path, e);
    }
    if (actual != expected) {
      throw new IllegalStateException(
          "%s declares %,d x %d (%,d bytes) but holds %,d bytes. A cropped big-ann download must have"
                  .formatted(path, shape.numPoints(), shape.dimensions(), expected, actual)
              + " its leading point count rewritten to match the prefix that was fetched.");
    }
    return readVectors(path, shape.numPoints(), shape.dimensions());
  }

  /** Reads the first {@code count} vectors, ignoring any larger count in the header. */
  public static float[][] readVectors(Path path, int count, int dimensions) {
    Shape shape = readShape(path);
    if (count < 0 || dimensions != shape.dimensions())
      throw new IllegalArgumentException(
          "prefix count must be nonnegative and dimension must match header");
    try {
      long required = HEADER_BYTES + (long) count * dimensions * Float.BYTES;
      if (count > shape.numPoints() || Files.size(path) < required)
        throw new IllegalStateException("vector prefix exceeds available data: " + path);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    int rowBytes = Math.multiplyExact(dimensions, Float.BYTES);
    float[][] out = new float[count][dimensions];
    byte[] row = new byte[rowBytes];
    try (InputStream raw = Files.newInputStream(path);
        BufferedInputStream in = new BufferedInputStream(raw, 1 << 20)) {
      if (in.skip(HEADER_BYTES) != HEADER_BYTES) {
        throw new IllegalStateException("could not skip the header of " + path);
      }
      ByteBuffer buffer = ByteBuffer.wrap(row).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < count; i++) {
        if (in.readNBytes(row, 0, rowBytes) != rowBytes)
          throw new java.io.EOFException("truncated vector row " + i + " in " + path);
        buffer.rewind();
        for (int d = 0; d < dimensions; d++) {
          out[i][d] = buffer.getFloat();
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException("reading vectors from " + path, e);
    }
    return out;
  }

  /**
   * Reads the neighbor-id block of a competition-format ground-truth file.
   *
   * <p>The trailing distances block is not read: recall@k compares id sets.
   */
  public static int[][] readGroundTruth(Path path) {
    try (InputStream raw = Files.newInputStream(path);
        BufferedInputStream in = new BufferedInputStream(raw, 1 << 20)) {
      DataInputStream header = new DataInputStream(in);
      int numQueries = readLeInt(header);
      int k = readLeInt(header);
      if (numQueries < 0 || k <= 0) throw new IOException("invalid ground truth header: " + path);
      if (Files.size(path) < HEADER_BYTES + (long) numQueries * k * Integer.BYTES)
        throw new java.io.EOFException("truncated ground truth ids: " + path);
      byte[] row = new byte[Math.multiplyExact(k, Integer.BYTES)];
      int[][] out = new int[numQueries][k];
      ByteBuffer buffer = ByteBuffer.wrap(row).order(ByteOrder.LITTLE_ENDIAN);
      for (int q = 0; q < numQueries; q++) {
        if (in.readNBytes(row, 0, row.length) != row.length)
          throw new java.io.EOFException("truncated ground truth row " + q + " in " + path);
        buffer.rewind();
        for (int i = 0; i < k; i++) {
          out[q][i] = buffer.getInt();
        }
      }
      return out;
    } catch (IOException e) {
      throw new UncheckedIOException("reading ground truth from " + path, e);
    }
  }

  private static int readLeInt(DataInputStream in) throws IOException {
    return Integer.reverseBytes(in.readInt());
  }
}
