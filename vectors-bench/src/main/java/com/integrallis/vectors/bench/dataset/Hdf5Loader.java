package com.integrallis.vectors.bench.dataset;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import java.nio.file.Path;

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

  private static float[][] readFloatMatrix(Path path, String datasetPath) {
    try (HdfFile hdf = new HdfFile(path.toFile())) {
      Dataset ds = hdf.getDatasetByPath(datasetPath);
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
