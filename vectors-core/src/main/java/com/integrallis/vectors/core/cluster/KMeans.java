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
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * K-means clustering with k-means++ seeding and Lloyd's iterations.
 *
 * <p>For datasets with {@code n > 100,000} rows, centroid update uses virtual threads partitioned
 * across available processors to parallelise the partial-sum accumulation.
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
    if (dataset.length == 0) throw new IllegalArgumentException("empty dataset");
    if (k <= 0 || k > dataset.length) throw new IllegalArgumentException("invalid k: " + k);
    int n = dataset.length;
    int dim = dataset[0].length;
    Random rng = new Random(seed);

    // k-means++ seeding
    float[][] centroids = new float[k][dim];
    centroids[0] = Arrays.copyOf(dataset[rng.nextInt(n)], dim);
    float[] minDist = new float[n];
    Arrays.fill(minDist, Float.MAX_VALUE);

    for (int c = 1; c < k; c++) {
      for (int i = 0; i < n; i++) {
        float d = VectorUtil.squareDistance(dataset[i], centroids[c - 1]);
        if (d < minDist[i]) minDist[i] = d;
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

    for (int iter = 0; iter < maxIter; iter++) {
      boolean changed = false;
      for (int i = 0; i < n; i++) {
        int nearest = nearestCentroid(dataset[i], centroids);
        if (nearest != assignments[i]) {
          assignments[i] = nearest;
          changed = true;
        }
      }
      if (!changed) break;
      if (useVirtualThreads) {
        updateCentroidsParallel(dataset, assignments, centroids, k, dim);
      } else {
        updateCentroids(dataset, assignments, centroids, k, dim);
      }
    }
    return centroids;
  }

  /** Assigns each vector in {@code dataset} to the index of its nearest centroid. */
  public static int[] assign(float[][] dataset, float[][] centroids) {
    int[] out = new int[dataset.length];
    for (int i = 0; i < dataset.length; i++) out[i] = nearestCentroid(dataset[i], centroids);
    return out;
  }

  /** Mean squared distance from each vector to its assigned centroid — lower is better. */
  public static double quantizationError(
      float[][] dataset, float[][] centroids, int[] assignments) {
    double total = 0.0;
    for (int i = 0; i < dataset.length; i++)
      total += VectorUtil.squareDistance(dataset[i], centroids[assignments[i]]);
    return total / dataset.length;
  }

  private static int nearestCentroid(float[] v, float[][] centroids) {
    int best = 0;
    float bestDist = VectorUtil.squareDistance(v, centroids[0]);
    for (int c = 1; c < centroids.length; c++) {
      float d = VectorUtil.squareDistance(v, centroids[c]);
      if (d < bestDist) {
        bestDist = d;
        best = c;
      }
    }
    return best;
  }

  private static void updateCentroids(
      float[][] dataset, int[] assignments, float[][] centroids, int k, int dim) {
    double[][] sums = new double[k][dim];
    int[] counts = new int[k];
    for (int i = 0; i < dataset.length; i++) {
      int c = assignments[i];
      counts[c]++;
      for (int d = 0; d < dim; d++) sums[c][d] += dataset[i][d];
    }
    for (int c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (int d = 0; d < dim; d++) centroids[c][d] = (float) (sums[c][d] / counts[c]);
      }
    }
  }

  private static void updateCentroidsParallel(
      float[][] dataset, int[] assignments, float[][] centroids, int k, int dim) {
    int n = dataset.length;
    int ncpu = Runtime.getRuntime().availableProcessors();
    int chunk = (n + ncpu - 1) / ncpu;
    double[][][] partialSums = new double[ncpu][k][dim];
    int[][] partialCounts = new int[ncpu][k];

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
}
