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
package com.integrallis.vectors.quantization;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
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

  /** Reads all float vectors from an .fvecs file. */
  static float[][] readFvecs(Path path) {
    var vectors = new ArrayList<float[]>();
    try (var dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      while (dis.available() > 0) {
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
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return vectors.toArray(float[][]::new);
  }

  /** Reads all integer vectors from an .ivecs file (typically ground truth neighbors). */
  static int[][] readIvecs(Path path) {
    var records = new ArrayList<int[]>();
    try (var dis = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
      while (dis.available() > 0) {
        int count = Integer.reverseBytes(dis.readInt());
        int[] neighbors = new int[count];
        for (int i = 0; i < count; i++) {
          neighbors[i] = Integer.reverseBytes(dis.readInt());
        }
        records.add(neighbors);
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return records.toArray(int[][]::new);
  }

  /** Wraps a float[][] as a {@link VectorDataset}. */
  static VectorDataset asDataset(float[][] vectors) {
    return new ArrayVectorDataset(vectors);
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
