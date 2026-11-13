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
import java.util.Arrays;
import java.util.Random;

/**
 * E-means clustering: a mini-batch k-means variant using exponential moving averages.
 *
 * <p>Algorithm from "E-Means: Scaling Laws for Mini-Batch K-Means" (Bessa et al., 2026). E-means
 * converges in ~5 epochs where Lloyd's k-means needs 25–50, with lower final loss and no additional
 * hyperparameters (batch size and step size are derived from the data via empirical scaling laws).
 *
 * <p>This is a faithful port of the FAISS NumPy CPU path (contrib/e_means/e_means.py), implementing
 * Algorithm 1 from the paper:
 *
 * <ol>
 *   <li>Sample initial centroids uniformly from the dataset.
 *   <li>For each epoch, shuffle the data and process random mini-batches.
 *   <li>Per batch: assign points to nearest centroids, accumulate sums and counts.
 *   <li>Per batch: apply the EMA update with cosine-annealed step size.
 *   <li>Every 5 updates: revive dead clusters by splitting high-mass donors.
 *   <li>After the last epoch: compute closing means from last-epoch assignments.
 * </ol>
 *
 * <p>Unlike Lloyd's full-batch k-means, the centroid update is incremental:
 *
 * <pre>
 *     N_new = gamma * N_old + alpha * n_j
 *     c_new = c_old + (alpha * n_j / N_new) * (mean_batch - c_old)
 * </pre>
 *
 * where alpha is cosine-annealed from alpha0 toward 0.1 * alpha0, and gamma = 1 - alpha.
 */
public final class EMeans {

  // -- Scaling-law constants (ported from FAISS contrib/e_means/base.py) --

  private static final double ALPHA_MIN_FACTOR = 0.1;
  private static final int CHECK_UNUSED_EVERY = 5;
  private static final float INIT_USAGE = (float) Math.pow(2.0, -16.0);
  private static final float SPLIT_EPSILON = 1e-4f;
  private static final float UNUSED_THRESHOLD = 0.1f;
  private static final int NO_REVIVAL_MAX_D = 2;
  private static final int AUTO_QUALITY_MAX_D = 16;

  private static final double C_B_THROUGHPUT = 0.035134;
  private static final double C_B_QUALITY = 0.00439175; // one eighth of throughput
  private static final double B_K = 0.134749;
  private static final double B_N = 0.876473;
  private static final double B_T = 0.813344;

  private static final double C_ALPHA = 0.8244;
  private static final double A_K = -0.4617;
  private static final double A_BATCH = 0.9003;
  private static final double A_N = -0.4498;
  private static final double A_T = -0.1057;

  private EMeans() {}

  /**
   * Train {@code k} centroids on {@code dataset} using the e-means algorithm.
   *
   * @param dataset training vectors [n][dim]; not modified
   * @param k number of centroids
   * @param nepochs number of passes over the dataset (fractional values rounded; default in FAISS
   *     is 5)
   * @param seed RNG seed for reproducibility
   * @return float[k][dim] centroid matrix
   */
  public static float[][] train(float[][] dataset, int k, float nepochs, long seed) {
    int n = validateMatrix(dataset, "dataset");
    int dim = dataset[0].length;
    if (k <= 0 || k > n) {
      throw new IllegalArgumentException("invalid k: " + k);
    }
    if (nepochs <= 0) {
      throw new IllegalArgumentException("nepochs must be > 0: " + nepochs);
    }

    // Resolve recipe (batch size and initial EMA step from scaling laws).
    String batchLaw = dim <= AUTO_QUALITY_MAX_D ? "quality" : "throughput";
    int batchSize = resolveBatchSize(n, k, nepochs, batchLaw);
    float alpha0 = resolveAlpha0(n, k, nepochs, batchSize);

    // Total logical updates across all epochs.
    int[] epochRows = resolveEpochRows(n, nepochs);
    int totalSteps = 0;
    for (int rows : epochRows) {
      totalSteps += (rows + batchSize - 1) / batchSize;
    }

    // Initialize centroids by sampling k distinct rows from the dataset.
    Random rng = new Random(seed);
    float[][] centroids = sampleCentroids(dataset, k, rng);

    // Running mass N_j for each centroid (Algorithm 1, line 1).
    float[] usage = new float[k];
    Arrays.fill(usage, INIT_USAGE);

    // Epoch accumulators for closing means (only last epoch is retained).
    double[][] passSums = null;
    double[] passCounts = null;

    int step = 0;
    int[] order = new int[n];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int epoch = 0; epoch < epochRows.length; epoch++) {
      int rowsThisEpoch = epochRows[epoch];
      boolean isLastEpoch = epoch == epochRows.length - 1;

      // Shuffle a fresh copy of indices for this epoch.
      shuffle(order, rng);

      if (isLastEpoch) {
        passSums = new double[k][dim];
        passCounts = new double[k];
      }

      // Process batches.
      for (int start = 0; start < rowsThisEpoch; start += batchSize) {
        int end = Math.min(start + batchSize, rowsThisEpoch);
        int batchLen = end - start;

        // Gather batch points into a contiguous array for SIMD batch scoring.
        float[][] batchPoints = new float[batchLen][dim];
        for (int b = start, i = 0; b < end; b++, i++) {
          batchPoints[i] = dataset[order[b]];
        }

        // Assignment: for each centroid, batch-score against all batch points via SIMD.
        int[] nearest = new int[batchLen];
        float[] bestDist = new float[batchLen];
        Arrays.fill(bestDist, Float.MAX_VALUE);
        float[] scores = new float[batchLen];

        // Accumulate batch: nearest-centroid assignment + sums/counts.
        double[][] batchSums = new double[k][dim];
        int[] batchCounts = new int[k];
        double batchObjective = 0.0;

        for (int j = 0; j < k; j++) {
          VectorUtil.batchSquaredL2(centroids[j], batchPoints, scores, batchLen);
          for (int i = 0; i < batchLen; i++) {
            if (scores[i] < bestDist[i]) {
              bestDist[i] = scores[i];
              nearest[i] = j;
            }
          }
        }

        for (int i = 0; i < batchLen; i++) {
          int idx = nearest[i];
          batchCounts[idx]++;
          float[] row = batchPoints[i];
          double[] sum = batchSums[idx];
          for (int d = 0; d < dim; d++) {
            sum[d] += row[d];
          }
          batchObjective += bestDist[i];
        }

        // Apply EMA update (Algorithm 1, line 8).
        float alpha = scheduledAlpha(alpha0, step, totalSteps);
        float decay = 1.0f - alpha;
        applyStatistics(
            centroids, usage, batchSums, batchCounts, batchObjective, alpha, decay, false);

        // If this is the last epoch, accumulate for closing means.
        if (isLastEpoch) {
          for (int c = 0; c < k; c++) {
            passCounts[c] += batchCounts[c];
            for (int d = 0; d < dim; d++) {
              passSums[c][d] += batchSums[c][d];
            }
          }
        }

        step++;

        // Revive dead clusters every CHECK_UNUSED_EVERY updates (Algorithm 1, line 9).
        if (step % CHECK_UNUSED_EVERY == 0 && dim > NO_REVIVAL_MAX_D) {
          reviveUnused(centroids, usage, rng);
        }
      }
    }

    // Finalize: compute closing means from last-epoch assignments.
    return finalize(centroids, passSums, passCounts);
  }

  // -- Recipe resolution (scaling laws from the paper) --

  private static int resolveBatchSize(int n, int k, float nepochs, String batchLaw) {
    double cB = "quality".equals(batchLaw) ? C_B_QUALITY : C_B_THROUGHPUT;
    double raw = cB * Math.pow(k, B_K) * Math.pow(n, B_N) * Math.pow(nepochs, B_T);
    int batch = roundPowerOfTwo(raw);
    // Cap to sensible bounds.
    if ("throughput".equals(batchLaw)) {
      batch = Math.min(batch, roundPowerOfTwo(n / 2.0));
    } else {
      batch = Math.min(batch, n / 2);
    }
    return Math.max(1, Math.min(batch, n));
  }

  private static float resolveAlpha0(int n, int k, float nepochs, int batchSize) {
    double alpha0 =
        C_ALPHA
            * Math.pow(k, A_K)
            * Math.pow(batchSize, A_BATCH)
            * Math.pow(n, A_N)
            * Math.pow(nepochs, A_T);
    return (float) Math.min(Math.max(alpha0, 1e-4), 1.0);
  }

  /** Cosine-anneal alpha toward ALPHA_MIN_FACTOR of alpha0. */
  private static float scheduledAlpha(float alpha0, int step, int totalSteps) {
    double fraction = Math.min(step, totalSteps) / Math.max(1.0, totalSteps);
    double cosine = 0.5 * (1.0 + Math.cos(Math.PI * fraction));
    return (float) (alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine));
  }

  /**
   * How many rows belong to each epoch (whole passes + optional fractional tail + closing pass).
   */
  private static int[] resolveEpochRows(int n, float nepochs) {
    int whole = (int) Math.floor(nepochs);
    int remainder = Math.round((nepochs - whole) * n);
    if (whole == 0) {
      return new int[] {remainder};
    }
    int[] rows = new int[whole + (remainder > 0 ? 1 : 0) + 1];
    int idx = 0;
    for (int i = 0; i < whole - 1; i++) {
      rows[idx++] = n;
    }
    if (remainder > 0) {
      rows[idx++] = remainder;
    }
    rows[idx] = n; // closing pass
    // Trim trailing zeros.
    if (rows[idx] == 0) {
      return Arrays.copyOf(rows, idx);
    }
    return Arrays.copyOf(rows, idx + 1);
  }

  // -- Core algorithm helpers --

  /** Apply the EMA update to centroids and running masses. */
  private static void applyStatistics(
      float[][] centroids,
      float[] usage,
      double[][] sums,
      int[] counts,
      double objective,
      float alpha,
      float decay,
      boolean replaceUsage) {

    int k = centroids.length;
    int dim = centroids[0].length;

    for (int c = 0; c < k; c++) {
      if (counts[c] == 0) {
        // Empty cluster: decay mass but leave position unchanged.
        if (!replaceUsage) {
          usage[c] *= decay;
        }
        continue;
      }

      if (!replaceUsage) {
        usage[c] *= decay;
      }
      float retained = replaceUsage ? 0.0f : usage[c];
      float mass = retained + alpha * counts[c];

      double[] mean = new double[dim];
      for (int d = 0; d < dim; d++) {
        mean[d] = sums[c][d] / counts[c];
      }

      float weight = alpha * counts[c] / mass;
      float[] pos = centroids[c];
      for (int d = 0; d < dim; d++) {
        float update = (float) (pos[d] + weight * (mean[d] - pos[d]));
        // When weight == 1, jump directly to the batch mean (avoids round-trip through old pos).
        if (weight == 1.0f) {
          update = (float) mean[d];
        }
        pos[d] = update;
      }

      usage[c] = mass;
    }
  }

  /** Revive dead clusters by splitting high-mass donors. */
  private static void reviveUnused(float[][] centroids, float[] usage, Random rng) {
    int k = centroids.length;
    int dim = centroids[0].length;

    float total = 0.0f;
    for (float u : usage) total += u;
    float threshold = total * UNUSED_THRESHOLD / k;

    int nDonors = 0;
    int nUnused = 0;
    for (int i = 0; i < k; i++) {
      if (usage[i] > threshold && usage[i] > 0) nDonors++;
      if (usage[i] < threshold) nUnused++;
    }
    int nSplit = Math.min(nUnused, nDonors);
    if (nSplit == 0) return;

    // Gumbel-max sampling: sample donors proportional to log(usage).
    double[] keys = new double[k];
    Arrays.fill(keys, Double.NEGATIVE_INFINITY);
    for (int i = 0; i < k; i++) {
      if (usage[i] > threshold && usage[i] > 0) {
        // log(usage) - log(-log(uniform))
        double uniform = Math.max(rng.nextDouble(), Double.MIN_VALUE);
        keys[i] = Math.log(usage[i]) - Math.log(-Math.log(uniform));
      }
    }
    int[] source = topIndices(keys, nSplit);

    // Select unused targets.
    int[] target = new int[nSplit];
    int t = 0;
    for (int i = 0; i < k && t < nSplit; i++) {
      if (usage[i] < threshold) {
        target[t++] = i;
      }
    }

    // Apply splits.
    for (int i = 0; i < nSplit; i++) {
      int sIdx = source[i];
      int tIdx = target[i];
      float[] donor = centroids[sIdx];
      float[] child = centroids[tIdx];

      // Create a slightly perturbed copy.
      for (int d = 0; d < dim; d++) {
        float sign = (d % 2 == 0) ? 1.0f : -1.0f;
        float eps = SPLIT_EPSILON * sign;
        child[d] = donor[d] * (1.0f + eps);
        donor[d] = donor[d] * (1.0f - eps);
      }

      // If donor and child are still identical (all coordinates were zero), perturb along
      // coordinate 0 using Math.nextAfter.
      boolean identical = true;
      for (int d = 0; d < dim; d++) {
        if (Float.compare(donor[d], child[d]) != 0) {
          identical = false;
          break;
        }
      }
      if (identical) {
        child[0] = Math.nextAfter(child[0], Float.POSITIVE_INFINITY);
      }

      float half = usage[sIdx] / 2.0f;
      usage[sIdx] = half;
      usage[tIdx] = half;
    }
  }

  /** Compute closing means from the last epoch's accumulated sums and counts. */
  private static float[][] finalize(float[][] centroids, double[][] passSums, double[] passCounts) {
    int k = centroids.length;
    int dim = centroids[0].length;
    float[][] result = new float[k][dim];
    for (int i = 0; i < k; i++) {
      if (passCounts[i] > 0) {
        for (int d = 0; d < dim; d++) {
          result[i][d] = (float) (passSums[i][d] / passCounts[i]);
        }
      } else {
        // Never assigned: keep the EMA position.
        result[i] = Arrays.copyOf(centroids[i], dim);
      }
    }
    return result;
  }

  // -- Utility helpers --

  private static float[][] sampleCentroids(float[][] dataset, int k, Random rng) {
    int n = dataset.length;
    int dim = dataset[0].length;
    float[][] centroids = new float[k][dim];
    if (n == k) {
      for (int i = 0; i < n; i++) centroids[i] = Arrays.copyOf(dataset[i], dim);
      return centroids;
    }
    // Randomly sample k distinct rows.
    int[] chosen = rng.ints(0, n).distinct().limit(k).toArray();
    for (int i = 0; i < k; i++) {
      centroids[i] = Arrays.copyOf(dataset[chosen[i]], dim);
    }
    return centroids;
  }

  /** Fisher-Yates shuffle. */
  private static void shuffle(int[] array, Random rng) {
    for (int i = array.length - 1; i > 0; i--) {
      int j = rng.nextInt(i + 1);
      int tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
    }
  }

  /** Round a positive value to the nearest power of two. */
  private static int roundPowerOfTwo(double value) {
    if (value <= 1.0) return 1;
    int low = 1 << (int) Math.floor(Math.log(value) / Math.log(2));
    int high = low << 1;
    return (value / low < high / value) ? low : high;
  }

  /** Return indices of the top-k largest values (descending order). */
  private static int[] topIndices(double[] values, int k) {
    int n = values.length;
    // Simple sort-based selection.
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; i++) indices[i] = i;
    Arrays.sort(indices, (a, b) -> Double.compare(values[b], values[a]));
    int[] result = new int[k];
    for (int i = 0; i < k; i++) result[i] = indices[i];
    return result;
  }

  private static int validateMatrix(float[][] m, String name) {
    if (m == null || m.length == 0) throw new IllegalArgumentException(name + " is empty");
    int dim = m[0].length;
    for (float[] row : m) {
      if (row == null || row.length != dim) {
        throw new IllegalArgumentException(name + " has ragged rows");
      }
    }
    return m.length;
  }
}
