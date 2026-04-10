package com.integrallis.vectors.vamana;

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
 * Reads SIFT-format vector files (.fvecs and .ivecs). Adapted from JVector's {@code
 * io.github.jbellis.jvector.example.util.SiftLoader}.
 *
 * <p>The .fvecs format stores vectors as: repeated [4-byte LE int dimension, dim * 4-byte LE
 * floats]. The .ivecs format stores integer vectors similarly: repeated [4-byte LE int count, count
 * * 4-byte LE ints].
 */
final class SiftLoader {

  private SiftLoader() {}

  /**
   * Reads all float vectors from an .fvecs file.
   *
   * <p>Uses {@link EOFException} to detect end-of-file rather than {@link
   * java.io.DataInputStream#available()}, which is unreliable for {@link BufferedInputStream}:
   * {@code available()} only reports bytes in the current buffer, not the remaining file length,
   * and can return 0 mid-file if the buffer is exhausted between reads.
   */
  static float[][] readFvecs(Path path) {
    var vectors = new ArrayList<float[]>();
    try (var dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      try {
        while (true) {
          int dimension = Integer.reverseBytes(dis.readInt());
          if (dimension <= 0 || dimension > 100_000) {
            throw new IOException(
                "Invalid dimension " + dimension + " in fvecs file (possible corruption)");
          }
          byte[] buffer = new byte[dimension * Float.BYTES];
          dis.readFully(buffer);
          ByteBuffer bb = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);
          float[] vector = new float[dimension];
          bb.asFloatBuffer().get(vector);
          vectors.add(vector);
        }
      } catch (EOFException ignored) {
        // Normal end of file — loop exits here.
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return vectors.toArray(float[][]::new);
  }

  /**
   * Reads all integer vectors from an .ivecs file (typically ground truth neighbors).
   *
   * <p>Uses {@link EOFException} to detect end-of-file; see {@link #readFvecs} for rationale.
   */
  static int[][] readIvecs(Path path) {
    var records = new ArrayList<int[]>();
    try (var dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      try {
        while (true) {
          int count = Integer.reverseBytes(dis.readInt());
          int[] neighbors = new int[count];
          for (int i = 0; i < count; i++) {
            neighbors[i] = Integer.reverseBytes(dis.readInt());
          }
          records.add(neighbors);
        }
      } catch (EOFException ignored) {
        // Normal end of file — loop exits here.
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return records.toArray(int[][]::new);
  }

  /** Computes recall@k: fraction of true top-k neighbors found in the approximate top-k. */
  static double recallAtK(int[] trueNeighbors, int[] approxNeighbors, int k) {
    int hits = 0;
    for (int i = 0; i < k && i < approxNeighbors.length; i++) {
      for (int j = 0; j < k && j < trueNeighbors.length; j++) {
        if (approxNeighbors[i] == trueNeighbors[j]) {
          hits++;
          break;
        }
      }
    }
    return (double) hits / k;
  }
}
