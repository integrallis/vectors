package com.integrallis.vectors.bench.dataset;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Reads the TexMex / BIGANN binary vector formats used by SIFT, GIST, Deep, and similar ANN
 * benchmark datasets.
 *
 * <p>Format specifications:
 *
 * <ul>
 *   <li><b>.fvecs</b> — repeated {@code [4-byte LE int dim] [dim × 4-byte LE float]}.
 *   <li><b>.ivecs</b> — repeated {@code [4-byte LE int count] [count × 4-byte LE int]}.
 *   <li><b>.bvecs</b> — repeated {@code [4-byte LE int dim] [dim × 1-byte unsigned int]}.
 * </ul>
 *
 * <p>All read methods use {@link EOFException} to detect end-of-file rather than {@link
 * DataInputStream#available()}, which reports only bytes in the current I/O buffer and can return 0
 * mid-file once a buffer boundary is crossed.
 *
 * <p>The returned arrays expose their internal storage directly (no defensive copy) for zero-copy
 * performance in benchmark setup code.
 */
public final class FvecsLoader {

  private FvecsLoader() {}

  /**
   * Reads all float vectors from an {@code .fvecs} file.
   *
   * @param path path to the {@code .fvecs} file
   * @return all vectors as a 2-D array {@code [vectorIndex][dimension]}
   * @throws UncheckedIOException if the file cannot be read or is malformed
   */
  public static float[][] readFvecs(Path path) {
    return readFvecs(path, Integer.MAX_VALUE);
  }

  /**
   * Reads up to {@code maxVectors} float vectors from an {@code .fvecs} file. Useful for loading a
   * subset of large datasets (e.g., the first 100 K vectors of SIFT-1M) without reading the entire
   * file into memory.
   *
   * @param path path to the {@code .fvecs} file
   * @param maxVectors maximum number of vectors to read; reads all if {@link Integer#MAX_VALUE}
   * @return vectors as a 2-D array; length ≤ {@code maxVectors}
   * @throws UncheckedIOException if the file cannot be read or is malformed
   */
  public static float[][] readFvecs(Path path, int maxVectors) {
    var vectors = new ArrayList<float[]>();
    try (var dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      try {
        while (vectors.size() < maxVectors) {
          int dimension = Integer.reverseBytes(dis.readInt());
          if (dimension <= 0 || dimension > 1_000_000) {
            throw new IOException("Invalid dimension " + dimension + " in fvecs file: " + path);
          }
          byte[] buffer = new byte[dimension * Float.BYTES];
          dis.readFully(buffer);
          ByteBuffer bb = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);
          float[] vector = new float[dimension];
          bb.asFloatBuffer().get(vector);
          vectors.add(vector);
        }
      } catch (EOFException ignored) {
        // Normal end-of-file.
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return vectors.toArray(float[][]::new);
  }

  /**
   * Reads all integer vectors from an {@code .ivecs} file (typically ground-truth neighbor lists).
   *
   * @param path path to the {@code .ivecs} file
   * @return all records as a 2-D array {@code [queryIndex][neighborOrdinals]}
   * @throws UncheckedIOException if the file cannot be read or is malformed
   */
  public static int[][] readIvecs(Path path) {
    var records = new ArrayList<int[]>();
    try (var dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      try {
        while (true) {
          int count = Integer.reverseBytes(dis.readInt());
          if (count <= 0 || count > 1_000_000) {
            throw new IOException("Invalid count " + count + " in ivecs file: " + path);
          }
          int[] neighbors = new int[count];
          for (int i = 0; i < count; i++) {
            neighbors[i] = Integer.reverseBytes(dis.readInt());
          }
          records.add(neighbors);
        }
      } catch (EOFException ignored) {
        // Normal end-of-file.
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return records.toArray(int[][]::new);
  }

  /**
   * Reads all byte vectors from a {@code .bvecs} file. Each byte is treated as an unsigned value in
   * the range [0, 255].
   *
   * @param path path to the {@code .bvecs} file
   * @return all vectors as a 2-D array {@code [vectorIndex][dimension]}; values are unsigned bytes
   *     stored in Java {@code byte} (i.e., values 128–255 appear as negative).
   * @throws UncheckedIOException if the file cannot be read or is malformed
   */
  public static byte[][] readBvecs(Path path) {
    var vectors = new ArrayList<byte[]>();
    try (var dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      try {
        while (true) {
          int dimension = Integer.reverseBytes(dis.readInt());
          if (dimension <= 0 || dimension > 1_000_000) {
            throw new IOException("Invalid dimension " + dimension + " in bvecs file: " + path);
          }
          byte[] vector = new byte[dimension];
          dis.readFully(vector);
          vectors.add(vector);
        }
      } catch (EOFException ignored) {
        // Normal end-of-file.
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return vectors.toArray(byte[][]::new);
  }
}
