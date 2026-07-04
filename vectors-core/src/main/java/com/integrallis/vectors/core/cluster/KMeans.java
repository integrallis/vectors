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
package com.integrallis.vectors.core.cluster;

import com.integrallis.vectors.core.VectorUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * K-means clustering with k-means++ seeding and Lloyd's iterations.
 *
 * <p>For datasets with {@code n > 100,000} rows, the Lloyd assignment step and centroid update both
 * use virtual threads partitioned across available processors, parallelising nearest-centroid
 * scoring and partial-sum accumulation respectively.
 *
 * <p>Nearest-centroid scoring uses the fused GEMV kernel {@link VectorUtil#batchSquaredL2} which
 * loads each point once and scores it against 4 centroids at a time, amortising point loads.
 */
public final class KMeans {

  private static final int VIRTUAL_THREAD_THRESHOLD = 100_000;

  private KMeans() {}

  /**
   * Trains {@code k} centroids on {@code dataset} using k-means++ seeding then Lloyd's iterations.
   *
   * @param dataset training vectors [n][dim]; not modified
   * @param k number of centroids
   * @param maxIter maximum Lloyd iterations (early exit on convergence)
   * @param seed RNG seed for reproducibility
   * @return float[k][dim] centroid matrix
   */
  public static float[][] train(float[][] dataset, int k, int maxIter, long seed) {
    int dim = validateMatrix(dataset, "dataset");
    if (k <= 0 || k > dataset.length) throw new IllegalArgumentException("invalid k: " + k);
    if (maxIter < 0) throw new IllegalArgumentException("maxIter must be >= 0: " + maxIter);
    int n = dataset.length;
    Random rng = new Random(seed);

    // k-means++ seeding
    float[][] centroids = new float[k][dim];
    centroids[0] = Arrays.copyOf(dataset[rng.nextInt(n)], dim);
    float[] minDist = new float[n];
    Arrays.fill(minDist, Float.MAX_VALUE);
    // Reusable buffer for the fused distance-to-newest-centroid pass over all points.
    float[] seedDist = new float[n];

    for (int c = 1; c < k; c++) {
      // Fused GEMV: squared L2 from the newest centroid to every point in one pass, amortising
      // the centroid load across 4 points at a time. Symmetric with squareDistance(point,
      // centroid).
      VectorUtil.batchSquaredL2(centroids[c - 1], dataset, seedDist, n);
      for (int i = 0; i < n; i++) {
        if (seedDist[i] < minDist[i]) minDist[i] = seedDist[i];
      }
      double total = 0.0;
      for (float d : minDist) total += d;
      double thresh = rng.nextDouble() * total;
      double cum = 0.0;
      int chosen = n - 1;
      for (int i = 0; i < n; i++) {
        cum += minDist[i];
        if (cum >= thresh) {
          chosen = i;
          break;
        }
      }
      centroids[c] = Arrays.copyOf(dataset[chosen], dim);
    }

    // Lloyd's iterations — initialise with sentinel so first pass always runs
    int[] assignments = new int[n];
    java.util.Arrays.fill(assignments, -1);
    boolean useVirtualThreads = n > VIRTUAL_THREAD_THRESHOLD;
    int ncpu = Runtime.getRuntime().availableProcessors();

    // Reusable per-iteration accumulators (cleared each iteration) to avoid re-allocation.
    double[][] sums;
    int[] counts;
    double[][][] partialSums;
    int[][] partialCounts;
    float[][] scratchPerThread;
    if (useVirtualThreads) {
      sums = null;
      counts = null;
      partialSums = new double[ncpu][k][dim];
      partialCounts = new int[ncpu][k];
      scratchPerThread = new float[ncpu][k];
    } else {
      sums = new double[k][dim];
      counts = new int[k];
      partialSums = null;
      partialCounts = null;
      scratchPerThread = null;
    }
    float[] scratch = useVirtualThreads ? null : new float[k];

    for (int iter = 0; iter < maxIter; iter++) {
      boolean changed =
          useVirtualThreads
              ? assignParallel(dataset, centroids, assignments, k, ncpu, scratchPerThread)
              : assignSerial(dataset, centroids, assignments, scratch);
      if (!changed) break;
      if (useVirtualThreads) {
        updateCentroidsParallel(
            dataset, assignments, centroids, k, dim, ncpu, partialSums, partialCounts);
      } else {
        updateCentroids(dataset, assignments, centroids, k, dim, sums, counts);
      }
    }
    return centroids;
  }

  /** Assigns each vector in {@code dataset} to the index of its nearest centroid. */
  public static int[] assign(float[][] dataset, float[][] centroids) {
    validateCompatibleMatrices(dataset, "dataset", centroids, "centroids");
    int[] out = new int[dataset.length];
    float[] scratch = new float[centroids.length];
    for (int i = 0; i < dataset.length; i++)
      out[i] = nearestCentroid(dataset[i], centroids, scratch);
    return out;
  }

  /** Mean squared distance from each vector to its assigned centroid — lower is better. */
  public static double quantizationError(
      float[][] dataset, float[][] centroids, int[] assignments) {
    validateCompatibleMatrices(dataset, "dataset", centroids, "centroids");
    Objects.requireNonNull(assignments, "assignments");
    if (assignments.length != dataset.length) {
      throw new IllegalArgumentException(
          "assignments.length must equal dataset.length: "
              + assignments.length
              + " != "
              + dataset.length);
    }
    double total = 0.0;
    for (int i = 0; i < dataset.length; i++) {
      if (assignments[i] < 0 || assignments[i] >= centroids.length) {
        throw new IllegalArgumentException(
            "assignment out of range at row " + i + ": " + assignments[i]);
      }
      total += VectorUtil.squareDistance(dataset[i], centroids[assignments[i]]);
    }
    return total / dataset.length;
  }

  /**
   * Serial Lloyd assignment pass over all rows. Returns {@code true} if any assignment changed.
   * Reuses the caller-supplied {@code scratch} buffer (length &ge; k) for the fused GEMV output.
   */
  private static boolean assignSerial(
      float[][] dataset, float[][] centroids, int[] assignments, float[] scratch) {
    boolean changed = false;
    for (int i = 0; i < dataset.length; i++) {
      int nearest = nearestCentroid(dataset[i], centroids, scratch);
      if (nearest != assignments[i]) {
        assignments[i] = nearest;
        changed = true;
      }
    }
    return changed;
  }

  /**
   * Parallel Lloyd assignment pass. Rows are partitioned into {@code ncpu} disjoint contiguous
   * chunks, one virtual thread per chunk. Writes to {@code assignments} are disjoint across
   * threads; {@code centroids} is read-only. Each thread owns a private scratch row so the fused
   * GEMV outputs never alias. The per-thread "changed" flags are OR-ed into a single result.
   */
  private static boolean assignParallel(
      float[][] dataset,
      float[][] centroids,
      int[] assignments,
      int k,
      int ncpu,
      float[][] scratchPerThread) {
    int n = dataset.length;
    int chunk = (n + ncpu - 1) / ncpu;
    AtomicBoolean changed = new AtomicBoolean(false);

    try (ExecutorService exec = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<?>> futures = new ArrayList<>(ncpu);
      for (int t = 0; t < ncpu; t++) {
        int start = t * chunk;
        int end = Math.min(start + chunk, n);
        float[] scratch = scratchPerThread[t];
        futures.add(
            exec.submit(
                () -> {
                  boolean local = false;
                  for (int i = start; i < end; i++) {
                    int nearest = nearestCentroid(dataset[i], centroids, scratch);
                    if (nearest != assignments[i]) {
                      assignments[i] = nearest;
                      local = true;
                    }
                  }
                  if (local) changed.set(true);
                }));
      }
      for (Future<?> f : futures) {
        try {
          f.get();
        } catch (Exception e) {
          throw new RuntimeException("virtual-thread assignment failed", e);
        }
      }
    }
    return changed.get();
  }

  /**
   * Returns the index of the nearest centroid to {@code v} using the fused {@link
   * VectorUtil#batchSquaredL2} kernel followed by an argmin over the {@code scratch} distances.
   * {@code scratch.length} must be &ge; {@code centroids.length}. Ties resolve to the lowest index,
   * matching a strict {@code <} scan.
   */
  private static int nearestCentroid(float[] v, float[][] centroids, float[] scratch) {
    int k = centroids.length;
    VectorUtil.batchSquaredL2(v, centroids, scratch, k);
    int best = 0;
    float bestDist = scratch[0];
    for (int c = 1; c < k; c++) {
      if (scratch[c] < bestDist) {
        bestDist = scratch[c];
        best = c;
      }
    }
    return best;
  }

  private static void updateCentroids(
      float[][] dataset,
      int[] assignments,
      float[][] centroids,
      int k,
      int dim,
      double[][] sums,
      int[] counts) {
    for (int c = 0; c < k; c++) {
      Arrays.fill(sums[c], 0.0);
      counts[c] = 0;
    }
    for (int i = 0; i < dataset.length; i++) {
      int c = assignments[i];
      counts[c]++;
      double[] sc = sums[c];
      float[] row = dataset[i];
      for (int d = 0; d < dim; d++) sc[d] += row[d];
    }
    for (int c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (int d = 0; d < dim; d++) centroids[c][d] = (float) (sums[c][d] / counts[c]);
      }
    }
  }

  private static void updateCentroidsParallel(
      float[][] dataset,
      int[] assignments,
      float[][] centroids,
      int k,
      int dim,
      int ncpu,
      double[][][] partialSums,
      int[][] partialCounts) {
    int n = dataset.length;
    int chunk = (n + ncpu - 1) / ncpu;
    for (int t = 0; t < ncpu; t++) {
      int[] pc = partialCounts[t];
      double[][] ps = partialSums[t];
      for (int c = 0; c < k; c++) {
        pc[c] = 0;
        Arrays.fill(ps[c], 0.0);
      }
    }

    try (ExecutorService exec = Executors.newVirtualThreadPerTaskExecutor()) {
      List<Future<?>> futures = new ArrayList<>(ncpu);
      for (int t = 0; t < ncpu; t++) {
        int start = t * chunk;
        int end = Math.min(start + chunk, n);
        int ti = t;
        futures.add(
            exec.submit(
                () -> {
                  for (int i = start; i < end; i++) {
                    int c = assignments[i];
                    partialCounts[ti][c]++;
                    for (int d = 0; d < dim; d++) partialSums[ti][c][d] += dataset[i][d];
                  }
                }));
      }
      for (Future<?> f : futures) {
        try {
          f.get();
        } catch (Exception e) {
          throw new RuntimeException("virtual-thread centroid update failed", e);
        }
      }
    }
    for (int c = 0; c < k; c++) {
      double[] sum = new double[dim];
      int count = 0;
      for (int t = 0; t < ncpu; t++) {
        count += partialCounts[t][c];
        for (int d = 0; d < dim; d++) sum[d] += partialSums[t][c][d];
      }
      if (count > 0) for (int d = 0; d < dim; d++) centroids[c][d] = (float) (sum[d] / count);
    }
  }

  private static void validateCompatibleMatrices(
      float[][] left, String leftName, float[][] right, String rightName) {
    int leftDim = validateMatrix(left, leftName);
    int rightDim = validateMatrix(right, rightName);
    if (leftDim != rightDim) {
      throw new IllegalArgumentException(
          leftName + " dimension differs from " + rightName + ": " + leftDim + " != " + rightDim);
    }
  }

  private static int validateMatrix(float[][] matrix, String name) {
    Objects.requireNonNull(matrix, name);
    if (matrix.length == 0) throw new IllegalArgumentException("empty " + name);
    float[] first = Objects.requireNonNull(matrix[0], name + "[0]");
    int dim = first.length;
    if (dim == 0) throw new IllegalArgumentException(name + " dimension must be > 0");
    for (int i = 1; i < matrix.length; i++) {
      float[] row = Objects.requireNonNull(matrix[i], name + "[" + i + "]");
      if (row.length != dim) {
        throw new IllegalArgumentException(
            name + " dimensions differ: row 0 has " + dim + " but row " + i + " has " + row.length);
      }
    }
    return dim;
  }
}
