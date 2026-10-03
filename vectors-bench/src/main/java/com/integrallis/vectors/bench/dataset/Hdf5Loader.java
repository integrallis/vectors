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

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Reads ANN-Benchmarks datasets stored in HDF5 format (GloVe, NYTimes, Deep, etc.).
 *
 * <p>ANN-Benchmarks HDF5 files contain four top-level datasets:
 *
 * <ul>
 *   <li>{@code /train} — base vectors, shape {@code [N, D]}, dtype float32.
 *   <li>{@code /test} — query vectors, shape {@code [Q, D]}, dtype float32.
 *   <li>{@code /neighbors} — ground-truth neighbor ordinals per query, shape {@code [Q, K]}, dtype
 *       int32.
 *   <li>{@code /distances} — ground-truth distances per query, shape {@code [Q, K]}, dtype float32
 *       (optional; not all ANN-Benchmarks datasets include this dataset).
 * </ul>
 *
 * <p>Example:
 *
 * <pre>{@code
 * Path hdf5 = DatasetRegistry.annBenchDataset("glove-100-angular");
 * float[][] train     = Hdf5Loader.readTrainVectors(hdf5);
 * float[][] test      = Hdf5Loader.readTestVectors(hdf5);
 * int[][]   neighbors = Hdf5Loader.readNeighbors(hdf5);
 * }</pre>
 *
 * <p>All returned arrays expose their internal storage directly for zero-copy benchmark setup.
 */
public final class Hdf5Loader {

  private Hdf5Loader() {}

  /**
   * Reads the {@code /train} dataset (base vectors) from an ANN-Benchmarks HDF5 file.
   *
   * @param path path to the {@code .hdf5} file
   * @return base vectors as a 2-D array {@code [vectorIndex][dimension]}
   */
  public static float[][] readTrainVectors(Path path) {
    return readFloatMatrix(path, "/train");
  }

  /**
   * Reads the {@code /test} dataset (query vectors) from an ANN-Benchmarks HDF5 file.
   *
   * @param path path to the {@code .hdf5} file
   * @return query vectors as a 2-D array {@code [queryIndex][dimension]}
   */
  public static float[][] readTestVectors(Path path) {
    return readFloatMatrix(path, "/test");
  }

  /**
   * Reads the {@code /neighbors} dataset (ground-truth neighbor ordinals) from an ANN-Benchmarks
   * HDF5 file. Each row contains the ordinals of the true nearest neighbors for the corresponding
   * query, sorted by ascending distance.
   *
   * @param path path to the {@code .hdf5} file
   * @return neighbor ordinals as a 2-D array {@code [queryIndex][neighborRank]}
   */
  public static int[][] readNeighbors(Path path) {
    return readIntMatrix(path, "/neighbors");
  }

  /**
   * Reads the {@code /distances} dataset from an ANN-Benchmarks HDF5 file. Distances correspond to
   * the neighbor ordinals returned by {@link #readNeighbors}.
   *
   * @param path path to the {@code .hdf5} file
   * @return distances as a 2-D array {@code [queryIndex][neighborRank]}
   */
  public static float[][] readDistances(Path path) {
    return readFloatMatrix(path, "/distances");
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  /**
   * Largest single read issued against an HDF5 dataset, in bytes.
   *
   * <p>A whole-dataset read maps the dataset into one buffer, and a {@link java.nio.ByteBuffer}
   * cannot exceed {@link Integer#MAX_VALUE} bytes. ANN-Benchmarks ships sets that are larger than
   * that: GIST1M's {@code /train} is 1,000,000 x 960 floats, 3.84 GB, and deep-image-96 is the same
   * order. Reading those used to fail with "Size exceeds Integer.MAX_VALUE" from inside the HDF5
   * library, which made two published datasets simply unavailable to this harness.
   *
   * <p>Kept well below the hard limit so that a row block never straddles it: 1 GB of a 2-D float
   * dataset is at least 260,000 rows even at 1,000 dimensions.
   */
  private static final long MAX_READ_BYTES = 1L << 30;

  private static float[][] readFloatMatrix(Path path, String datasetPath) {
    try (HdfFile hdf = new HdfFile(path.toFile())) {
      Dataset ds = hdf.getDatasetByPath(datasetPath);
      if (ds.getSizeInBytes() > MAX_READ_BYTES) {
        return readFloatMatrixInBlocks(ds, datasetPath);
      }
      Object raw = ds.getData();
      if (raw instanceof float[][] matrix) {
        return matrix;
      }
      // jhdf may return a linearized float[] for certain HDF5 chunked layouts.
      if (raw instanceof float[] flat) {
        // getDimensions() returns int[] in jhdf 0.9.x
        int[] dims = ds.getDimensions();
        if (dims.length != 2) {
          throw new IllegalStateException(
              "Expected 2-D dataset at " + datasetPath + " but got " + dims.length + " dims");
        }
        return reshape(flat, dims[0], dims[1]);
      }
      throw new IllegalStateException(
          "Unexpected data type for " + datasetPath + ": " + raw.getClass().getName());
    }
  }

  /**
   * Reads a 2-D float dataset in row blocks, each under {@link #MAX_READ_BYTES}.
   *
   * <p>Uses the sliced form of the HDF5 read, so no single call maps more than one block. The
   * result is assembled into the same {@code float[rows][cols]} shape a whole-dataset read would
   * produce, which keeps every caller unchanged.
   */
  private static float[][] readFloatMatrixInBlocks(Dataset ds, String datasetPath) {
    int[] dims = ds.getDimensions();
    if (dims.length != 2) {
      throw new IllegalStateException(
          "Expected 2-D dataset at " + datasetPath + " but got " + dims.length + " dims");
    }
    int rows = dims[0];
    int cols = dims[1];
    long rowBytes = (long) cols * Float.BYTES;
    int blockRows = (int) Math.max(1, Math.min(rows, MAX_READ_BYTES / Math.max(1, rowBytes)));

    float[][] out = new float[rows][];
    for (int start = 0; start < rows; start += blockRows) {
      int count = Math.min(blockRows, rows - start);
      Object raw = ds.getData(new long[] {start, 0}, new int[] {count, cols});
      if (raw instanceof float[][] block) {
        System.arraycopy(block, 0, out, start, count);
      } else if (raw instanceof float[] flat) {
        for (int r = 0; r < count; r++) {
          out[start + r] = Arrays.copyOfRange(flat, r * cols, (r + 1) * cols);
        }
      } else {
        throw new IllegalStateException(
            "Unexpected data type for a block of " + datasetPath + ": " + raw.getClass().getName());
      }
    }
    return out;
  }

  private static int[][] readIntMatrix(Path path, String datasetPath) {
    try (HdfFile hdf = new HdfFile(path.toFile())) {
      Dataset ds = hdf.getDatasetByPath(datasetPath);
      Object raw = ds.getData();
      if (raw instanceof int[][] matrix) {
        return matrix;
      }
      if (raw instanceof int[] flat) {
        int[] dims = ds.getDimensions();
        if (dims.length != 2) {
          throw new IllegalStateException(
              "Expected 2-D dataset at " + datasetPath + " but got " + dims.length + " dims");
        }
        return reshapeInt(flat, dims[0], dims[1]);
      }
      throw new IllegalStateException(
          "Unexpected data type for " + datasetPath + ": " + raw.getClass().getName());
    }
  }

  private static float[][] reshape(float[] flat, int rows, int cols) {
    float[][] out = new float[rows][cols];
    for (int r = 0; r < rows; r++) {
      System.arraycopy(flat, r * cols, out[r], 0, cols);
    }
    return out;
  }

  private static int[][] reshapeInt(int[] flat, int rows, int cols) {
    int[][] out = new int[rows][cols];
    for (int r = 0; r < rows; r++) {
      System.arraycopy(flat, r * cols, out[r], 0, cols);
    }
    return out;
  }
}
