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

import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Product Quantization (PQ) of float32 vectors into compact byte codes.
 *
 * <p>Splits a D-dimensional vector into M sub-vectors and independently quantizes each sub-vector
 * using a learned codebook of Ks centroids. Each encoded vector is M bytes (one cluster index per
 * subspace), achieving D*4/M compression ratio vs float32.
 *
 * <p>Follows the JVector/FAISS approach:
 *
 * <ul>
 *   <li>Sub-vector decomposition with support for uneven splits (when D % M != 0)
 *   <li>K-means++ clustering per subspace (default 256 centroids)
 *   <li>Optional global centroid subtraction before quantization
 *   <li>ADC (Asymmetric Distance Computation) scoring via precomputed lookup tables
 * </ul>
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * ProductQuantizer pq = ProductQuantizer.train(data, 16);          // 16 subspaces
 * PQVectors compressed = pq.encodeAll(data);
 * ScoreFunction scorer = compressed.scoreFunctionFor(query, SimilarityFunction.DOT_PRODUCT);
 * }</pre>
 */
public final class ProductQuantizer implements Quantizer<PQVectors> {

  /** Default number of clusters per subspace (one byte index = 256 values). */
  static final int DEFAULT_CLUSTERS = 256;

  /** Maximum number of training vectors sampled for k-means. */
  static final int MAX_PQ_TRAINING_SET_SIZE = 25_000;

  private final int dimension;
  private final int numSubspaces; // M
  private final int numClusters; // Ks
  private final int[][] subspaceSizesAndOffsets; // [m][0]=size, [m][1]=offset
  private final float[][] codebooks; // codebooks[m] has size numClusters * subDim_m
  private final float[] globalCentroid; // null if not centered
  // Anisotropic PQ (ScaNN / AVQ): threshold >= 0 switches encode() to the coordinate-descent
  // encoder that minimises the weighted MIPS-error cost. UNWEIGHTED (-1) keeps the legacy
  // nearest-L2 encoder so byte-identical output is preserved for unweighted PQ.
  private final float anisotropicThreshold;
  // centroidNormsSquared[m][c] = ||c||^2 for centroid c in subspace m. Precomputed once in the
  // constructor — used by the anisotropic encoder to build per-(subspace, cluster) residuals in
  // O(1) per centroid (excluding the one dot product).
  private final float[][] centroidNormsSquared;

  private ProductQuantizer(
      int dimension,
      int numSubspaces,
      int numClusters,
      int[][] subspaceSizesAndOffsets,
      float[][] codebooks,
      float[] globalCentroid,
      float anisotropicThreshold) {
    this.dimension = dimension;
    this.numSubspaces = numSubspaces;
    this.numClusters = numClusters;
    this.subspaceSizesAndOffsets = subspaceSizesAndOffsets;
    this.codebooks = codebooks;
    this.globalCentroid = globalCentroid;
    this.anisotropicThreshold = anisotropicThreshold;
    this.centroidNormsSquared =
        anisotropicThreshold >= 0f ? precomputeCentroidNormsSquared(codebooks, numClusters) : null;
  }

  private static float[][] precomputeCentroidNormsSquared(float[][] codebooks, int numClusters) {
    float[][] norms = new float[codebooks.length][numClusters];
    for (int m = 0; m < codebooks.length; m++) {
      float[] book = codebooks[m];
      int subDim = book.length / numClusters;
      for (int c = 0; c < numClusters; c++) {
        int off = c * subDim;
        float s = 0f;
        for (int i = 0; i < subDim; i++) s += book[off + i] * book[off + i];
        norms[m][c] = s;
      }
    }
    return norms;
  }

  // --- Factory methods ---

  /**
   * Reconstructs a trained {@code ProductQuantizer} from previously serialized state. Used by
   * deserialization codecs to restore a quantizer without re-running k-means++ training.
   *
   * @param dimension the original vector dimension
   * @param numSubspaces number of sub-vector partitions (M)
   * @param numClusters centroids per subspace (Ks)
   * @param subspaceSizesAndOffsets per-subspace [size, offset] pairs
   * @param codebooks per-subspace codebook arrays (flat float arrays of size Ks * subDim)
   * @param globalCentroid the global centroid, or null if centering was not enabled
   * @return a reconstructed product quantizer
   */
  public static ProductQuantizer fromState(
      int dimension,
      int numSubspaces,
      int numClusters,
      int[][] subspaceSizesAndOffsets,
      float[][] codebooks,
      float[] globalCentroid) {
    return fromState(
        dimension,
        numSubspaces,
        numClusters,
        subspaceSizesAndOffsets,
        codebooks,
        globalCentroid,
        KMeansPlusPlusClusterer.UNWEIGHTED);
  }

  /**
   * Same as {@link #fromState(int, int, int, int[][], float[][], float[])} with the anisotropic
   * threshold that was used at training time. Pass {@link KMeansPlusPlusClusterer#UNWEIGHTED} for
   * standard PQ. When {@code anisotropicThreshold >= 0}, {@link #encode} uses the coordinate-
   * descent encoder instead of nearest-L2.
   */
  public static ProductQuantizer fromState(
      int dimension,
      int numSubspaces,
      int numClusters,
      int[][] subspaceSizesAndOffsets,
      float[][] codebooks,
      float[] globalCentroid,
      float anisotropicThreshold) {
    return new ProductQuantizer(
        dimension,
        numSubspaces,
        numClusters,
        subspaceSizesAndOffsets,
        codebooks,
        globalCentroid,
        anisotropicThreshold);
  }

  /**
   * Trains a PQ with default 256 clusters and global centering enabled.
   *
   * @param dataset the training data
   * @param numSubspaces number of sub-vector partitions (M)
   * @return a trained product quantizer
   */
  public static ProductQuantizer train(VectorDataset dataset, int numSubspaces) {
    return train(dataset, numSubspaces, DEFAULT_CLUSTERS, true);
  }

  /**
   * Trains a PQ with the given cluster count and global centering enabled.
   *
   * @param dataset the training data
   * @param numSubspaces number of sub-vector partitions (M)
   * @param numClusters centroids per subspace (Ks)
   * @return a trained product quantizer
   */
  public static ProductQuantizer train(VectorDataset dataset, int numSubspaces, int numClusters) {
    return train(dataset, numSubspaces, numClusters, true);
  }

  /**
   * Trains a PQ with full control over parameters (single-threaded, deterministic). Equivalent to
   * {@link #train(VectorDataset, int, int, boolean, int)} with {@code trainThreads == 1}.
   *
   * @param dataset the training data
   * @param numSubspaces number of sub-vector partitions (M)
   * @param numClusters centroids per subspace (Ks), must be &le; 256
   * @param center if true, subtract global centroid before quantization
   * @return a trained product quantizer
   */
  public static ProductQuantizer train(
      VectorDataset dataset, int numSubspaces, int numClusters, boolean center) {
    return train(dataset, numSubspaces, numClusters, center, 1);
  }

  /**
   * Trains a PQ with full control over parameters and optional per-subspace parallelism.
   *
   * <p>The outer loop over M subspaces is trivially embarrassingly parallel — each subspace
   * clusters an independent slice of the training data into an independent codebook. Passing {@code
   * trainThreads > 1} schedules the M k-means jobs on a fixed-size {@link ExecutorService}.
   *
   * <p>Determinism: two calls with the same parameters always return identical codebooks. When
   * {@code trainThreads == 1}, the sequential legacy path is preserved (shared {@link Random}
   * chained across subspaces, seed {@value #SEED}) so output is byte-identical to pre-R2.E
   * releases. When {@code trainThreads > 1}, each subspace derives its own seed deterministically
   * from ({@value #SEED}, subspace index), yielding a different — but reproducible — codebook.
   *
   * @param dataset the training data
   * @param numSubspaces number of sub-vector partitions (M)
   * @param numClusters centroids per subspace (Ks), must be &le; 256
   * @param center if true, subtract global centroid before quantization
   * @param trainThreads worker thread count for per-subspace k-means (must be {@code >= 1}); set to
   *     {@code Math.min(M, Runtime.getRuntime().availableProcessors())} for typical use
   * @return a trained product quantizer
   */
  public static ProductQuantizer train(
      VectorDataset dataset, int numSubspaces, int numClusters, boolean center, int trainThreads) {
    return train(
        dataset,
        numSubspaces,
        numClusters,
        center,
        trainThreads,
        KMeansPlusPlusClusterer.UNWEIGHTED);
  }

  /**
   * Same as {@link #train(VectorDataset, int, int, boolean, int)} with an optional anisotropic
   * refinement threshold. Pass {@link KMeansPlusPlusClusterer#UNWEIGHTED} to disable (standard PQ).
   *
   * <p>Valid threshold range is {@code [0, 1)}; typical values are 0.1\u20130.3. Larger values
   * sharpen the parallel-direction weighting. See <em>Guo et al. 2020</em> (ScaNN / AVQ) \u00a73.
   */
  public static ProductQuantizer train(
      VectorDataset dataset,
      int numSubspaces,
      int numClusters,
      boolean center,
      int trainThreads,
      float anisotropicThreshold) {
    if (trainThreads < 1) {
      throw new IllegalArgumentException("trainThreads must be >= 1: " + trainThreads);
    }
    int dim = dataset.dimension();
    validateParameters(dim, numSubspaces, numClusters);

    int[][] sizesAndOffsets = computeSubspaceSizesAndOffsets(dim, numSubspaces);

    // Optionally compute and subtract global centroid
    float[] centroid = center ? dataset.computeCentroid() : null;

    // Sample training data if dataset is large
    float[][] trainingData = sampleTrainingData(dataset, centroid);

    float[][] codebooks = new float[numSubspaces][];
    int anisoIters =
        anisotropicThreshold >= 0f ? KMeansPlusPlusClusterer.DEFAULT_MAX_ITERATIONS : 0;

    if (trainThreads == 1 || numSubspaces == 1) {
      // Legacy sequential path: a single Random chained across all subspaces. Preserves
      // byte-identical codebook output vs pre-R2.E releases.
      Random rng = new Random(SEED);
      for (int m = 0; m < numSubspaces; m++) {
        int subDim = sizesAndOffsets[m][0];
        int offset = sizesAndOffsets[m][1];
        float[] subVectors = extractSubvectors(trainingData, offset, subDim);
        codebooks[m] =
            KMeansPlusPlusClusterer.cluster(
                subVectors,
                trainingData.length,
                subDim,
                numClusters,
                KMeansPlusPlusClusterer.DEFAULT_MAX_ITERATIONS,
                anisoIters,
                anisotropicThreshold,
                rng);
      }
    } else {
      // Parallel path: per-subspace deterministic seeds so output is reproducible across thread
      // counts and JDK scheduler variance. Codebooks will differ numerically from the sequential
      // path but remain algorithmically equivalent (identical MSE-convergence properties).
      int workers = Math.min(trainThreads, numSubspaces);
      ExecutorService pool =
          Executors.newFixedThreadPool(
              workers,
              r -> {
                Thread t = new Thread(r, "pq-train-" + PQ_TRAIN_WORKER_ID.incrementAndGet());
                t.setDaemon(true);
                return t;
              });
      try {
        @SuppressWarnings({"unchecked", "rawtypes"})
        Future<float[]>[] futures = new Future[numSubspaces];
        for (int m = 0; m < numSubspaces; m++) {
          final int mi = m;
          final int[] sz = sizesAndOffsets[m];
          final int aIters = anisoIters;
          final float aT = anisotropicThreshold;
          futures[m] =
              pool.submit(
                  () -> clusterSubspaceIndependent(trainingData, sz, numClusters, mi, aIters, aT));
        }
        for (int m = 0; m < numSubspaces; m++) {
          try {
            codebooks[m] = futures[m].get();
          } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("PQ training interrupted at subspace " + m, ie);
          } catch (ExecutionException ee) {
            Throwable cause = ee.getCause();
            if (cause instanceof RuntimeException re) throw re;
            if (cause instanceof Error err) throw err;
            throw new RuntimeException("PQ training failed at subspace " + m, cause);
          }
        }
      } finally {
        pool.shutdown();
        try {
          if (!pool.awaitTermination(5, TimeUnit.MINUTES)) {
            pool.shutdownNow();
          }
        } catch (InterruptedException ie) {
          pool.shutdownNow();
          Thread.currentThread().interrupt();
        }
      }
    }

    return new ProductQuantizer(
        dim, numSubspaces, numClusters, sizesAndOffsets, codebooks, centroid, anisotropicThreshold);
  }

  /** Deterministic base seed shared by all train() overloads. */
  private static final long SEED = 42L;

  /** SplitMix64-style mixer for derived per-subspace seeds (reproducible across JDK versions). */
  private static final long SEED_MIXER = 0x9E3779B97F4A7C15L;

  /** Counter used to give pq-train worker threads unique names across invocations. */
  private static final AtomicInteger PQ_TRAIN_WORKER_ID = new AtomicInteger();

  /**
   * Clusters one subspace with an independent, per-subspace-seeded {@link Random}. Used only on the
   * parallel train path; thread-safe because it reads {@code trainingData} and writes only a
   * freshly-allocated return array.
   */
  private static float[] clusterSubspaceIndependent(
      float[][] trainingData,
      int[] sizeAndOffset,
      int numClusters,
      int m,
      int anisoIters,
      float anisoT) {
    int subDim = sizeAndOffset[0];
    int offset = sizeAndOffset[1];
    float[] subVectors = extractSubvectors(trainingData, offset, subDim);
    Random rng = new Random(SEED ^ ((long) m * SEED_MIXER));
    return KMeansPlusPlusClusterer.cluster(
        subVectors,
        trainingData.length,
        subDim,
        numClusters,
        KMeansPlusPlusClusterer.DEFAULT_MAX_ITERATIONS,
        anisoIters,
        anisoT,
        rng);
  }

  // --- Quantizer interface ---

  @Override
  public byte[] encode(float[] vector) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + ", got " + vector.length);
    }
    byte[] code = new byte[numSubspaces];
    encodeInto(vector, code);
    return code;
  }

  @Override
  public float encode(float[] vector, byte[] dst) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + ", got " + vector.length);
    }
    encodeInto(vector, dst);
    return 0f; // PQ has no per-vector correction factor
  }

  @Override
  public float[] decode(byte[] encoded) {
    float[] result = new float[dimension];

    for (int m = 0; m < numSubspaces; m++) {
      int subDim = subspaceSizesAndOffsets[m][0];
      int offset = subspaceSizesAndOffsets[m][1];
      int clusterIdx = encoded[m] & 0xFF;
      System.arraycopy(codebooks[m], clusterIdx * subDim, result, offset, subDim);
    }

    // Add back global centroid if it was subtracted
    if (globalCentroid != null) {
      VectorUtil.addInPlace(result, globalCentroid);
    }

    return result;
  }

  @Override
  public PQVectors encodeAll(VectorDataset dataset) {
    int count = dataset.size();
    byte[][] codes = new byte[count][numSubspaces];
    for (int i = 0; i < count; i++) {
      encodeInto(dataset.getVector(i), codes[i]);
    }
    return new PQVectors(this, codes, dataset.dimension());
  }

  @Override
  public float compressionRatio() {
    // Original: dimension * 4 bytes (float32)
    // Compressed: numSubspaces * 1 byte
    return (dimension * 4.0f) / numSubspaces;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // --- Accessors ---

  /** Returns the number of sub-vector partitions (M). */
  public int numSubspaces() {
    return numSubspaces;
  }

  /** Returns the number of centroids per subspace (Ks). */
  public int numClusters() {
    return numClusters;
  }

  /**
   * Returns the anisotropic threshold used at training time. {@link
   * KMeansPlusPlusClusterer#UNWEIGHTED} (-1) means standard unweighted PQ (nearest-L2 encoder);
   * {@code >= 0} means the coordinate-descent anisotropic encoder is active. Exposed primarily for
   * codecs that need to round-trip the value through serialisation.
   */
  public float anisotropicThreshold() {
    return anisotropicThreshold;
  }

  /**
   * Returns the codebook for subspace m. The codebook is a flat array of size {@code numClusters *
   * subDim}, where entry c starts at index {@code c * subDim}.
   */
  public float[] codebook(int m) {
    return codebooks[m];
  }

  /** Returns all codebooks as a 2D array, one per subspace. */
  public float[][] codebooks() {
    return codebooks;
  }

  /** Returns the subspace size and offset as [size, offset]. */
  public int[] subspaceSizeAndOffset(int m) {
    return subspaceSizesAndOffsets[m];
  }

  /** Returns all subspace sizes and offsets as a 2D array. */
  public int[][] subspaceSizesAndOffsets() {
    return subspaceSizesAndOffsets;
  }

  /** Returns the global centroid, or null if centering is not enabled. */
  public float[] globalCentroid() {
    return globalCentroid;
  }

  // --- ADC lookup table computation ---

  /**
   * Builds the ADC (Asymmetric Distance Computation) lookup table for a query. Returns a 2D table
   * where {@code table[m][k]} is the partial distance/similarity between query sub-vector m and
   * centroid k.
   *
   * <p>This method is public so that external callers (e.g., fused ADC graph scorers in {@code
   * vectors-hnsw}) can build the table once per query and score many encoded vectors via O(M) table
   * lookups instead of O(dim) float operations.
   *
   * @param query the query vector (full precision, before centering)
   * @param useDotProduct true to fill with dot products, false for squared L2 distances
   * @return lookup table of size [numSubspaces][numClusters]
   */
  public float[][] buildADCTable(float[] query, boolean useDotProduct) {
    float[] q = query;
    if (globalCentroid != null) {
      q = Arrays.copyOf(query, query.length);
      VectorUtil.subInPlace(q, globalCentroid);
    }

    float[][] table = new float[numSubspaces][numClusters];

    for (int m = 0; m < numSubspaces; m++) {
      int subDim = subspaceSizesAndOffsets[m][0];
      int offset = subspaceSizesAndOffsets[m][1];
      float[] codebook = codebooks[m];

      for (int k = 0; k < numClusters; k++) {
        table[m][k] =
            useDotProduct
                ? VectorUtil.dotProduct(q, offset, codebook, k * subDim, subDim)
                : VectorUtil.squareDistance(q, offset, codebook, k * subDim, subDim);
      }
    }

    return table;
  }

  // --- Internal methods ---

  private void encodeInto(float[] vector, byte[] dst) {
    float[] v = vector;
    if (globalCentroid != null) {
      v = Arrays.copyOf(vector, vector.length);
      VectorUtil.subInPlace(v, globalCentroid);
    }

    if (anisotropicThreshold >= 0f) {
      encodeAnisotropic(v, dst);
      return;
    }

    for (int m = 0; m < numSubspaces; m++) {
      int subDim = subspaceSizesAndOffsets[m][0];
      int offset = subspaceSizesAndOffsets[m][1];
      float[] codebook = codebooks[m];

      // Find nearest centroid for this subspace
      int nearest = 0;
      float nearestDist = VectorUtil.squareDistance(v, offset, codebook, 0, subDim);

      for (int k = 1; k < numClusters; k++) {
        float dist = VectorUtil.squareDistance(v, offset, codebook, k * subDim, subDim);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearest = k;
        }
      }

      dst[m] = (byte) nearest;
    }
  }

  /**
   * Coordinate-descent anisotropic encoder (ScaNN / AVQ — Guo et al. 2020). Port of JVector's
   * {@code encodeAnisotropic}; uses the same parallel-cost-multiplier and residual decomposition as
   * {@link KMeansPlusPlusClusterer#parallelCostMultiplier} so the encoded codes are consistent with
   * the codebooks produced by anisotropic k-means.
   *
   * <p>After training with an anisotropic loss, the centroids are shaped to preserve parallel
   * components of each training point. Encoding with plain nearest-L2 would then select centroids
   * that match the overall point norm rather than its parallel component, erasing most of the
   * training benefit. This encoder swaps each subspace's centroid in turn whenever the total
   * weighted cost {@code pcm * ||e_‖||² + ||e_⊥||²} decreases.
   */
  private void encodeAnisotropic(float[] v, byte[] dst) {
    // Full-vector inverse norm (for the parallel-residual component).
    float xNormSq = 0f;
    for (float x : v) xNormSq += x * x;
    float inverseNorm = xNormSq > 0f ? (float) (1.0 / Math.sqrt(xNormSq)) : 0f;
    float pcm = KMeansPlusPlusClusterer.parallelCostMultiplier(anisotropicThreshold, dimension);

    // Pre-compute residuals[m][k] = (||x_m - c_{m,k}||², (c_{m,k}·x_m - ||x_m||²)² * inverseNorm)
    // and initialise dst[m] to the centroid with minimum residual norm squared (same as standard
    // nearest-L2 encoding) — this gives the coordinate descent a good starting point.
    float[][] residualNormSq = new float[numSubspaces][numClusters];
    float[][] parallelComp = new float[numSubspaces][numClusters];
    float parallelSum = 0f;

    for (int m = 0; m < numSubspaces; m++) {
      int subDim = subspaceSizesAndOffsets[m][0];
      int offset = subspaceSizesAndOffsets[m][1];
      float[] book = codebooks[m];
      float[] cNorms = centroidNormsSquared[m];
      // Subvector norm squared.
      float subNormSq = 0f;
      for (int i = 0; i < subDim; i++) subNormSq += v[offset + i] * v[offset + i];

      int bestIdx = 0;
      float bestNormSq = Float.POSITIVE_INFINITY;
      for (int c = 0; c < numClusters; c++) {
        float cDotX = VectorUtil.dotProduct(v, offset, book, c * subDim, subDim);
        float normSq = cNorms[c] - 2f * cDotX + subNormSq;
        float parallelErrorSubtotal = cDotX - subNormSq;
        residualNormSq[m][c] = normSq;
        parallelComp[m][c] = parallelErrorSubtotal * parallelErrorSubtotal * inverseNorm;
        if (normSq < bestNormSq) {
          bestNormSq = normSq;
          bestIdx = c;
        }
      }
      dst[m] = (byte) bestIdx;
      parallelSum += parallelComp[m][bestIdx];
    }

    // Coordinate descent: at most 10 sweeps, breaking out once a full pass makes no change.
    // Loop count matches JVector / SCANN reference.
    final int MAX_ITERATIONS = 10;
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
      boolean changed = false;
      for (int m = 0; m < numSubspaces; m++) {
        int oldIdx = dst[m] & 0xFF;
        float oldNormSq = residualNormSq[m][oldIdx];
        float oldParallel = parallelComp[m][oldIdx];
        float bestCostDelta = 0f;
        int bestIdx = oldIdx;
        float bestParallelSum = parallelSum;
        for (int c = 0; c < numClusters; c++) {
          if (c == oldIdx) continue;
          float thisParallelSum = parallelSum - oldParallel + parallelComp[m][c];
          float parallelNormDelta = thisParallelSum * thisParallelSum - parallelSum * parallelSum;
          if (parallelNormDelta > 0f) continue; // parallel cost got worse
          float residualNormDelta = residualNormSq[m][c] - oldNormSq;
          float perpendicularNormDelta = residualNormDelta - parallelNormDelta;
          float costDelta = pcm * parallelNormDelta + perpendicularNormDelta;
          if (costDelta < bestCostDelta) {
            bestCostDelta = costDelta;
            bestIdx = c;
            bestParallelSum = thisParallelSum;
          }
        }
        if (bestIdx != oldIdx) {
          dst[m] = (byte) bestIdx;
          parallelSum = bestParallelSum;
          changed = true;
        }
      }
      if (!changed) break;
    }
  }

  private static void validateParameters(int dim, int numSubspaces, int numClusters) {
    if (numSubspaces <= 0) {
      throw new IllegalArgumentException("numSubspaces must be > 0: " + numSubspaces);
    }
    if (numSubspaces > dim) {
      throw new IllegalArgumentException(
          "numSubspaces (" + numSubspaces + ") must be <= dimension (" + dim + ")");
    }
    if (numClusters <= 0 || numClusters > 256) {
      throw new IllegalArgumentException("numClusters must be in [1, 256]: " + numClusters);
    }
  }

  /**
   * Computes subspace sizes and offsets. When D % M != 0, the first (D % M) subspaces get one extra
   * dimension (JVector pattern).
   */
  static int[][] computeSubspaceSizesAndOffsets(int dim, int numSubspaces) {
    int[][] result = new int[numSubspaces][2];
    int baseSize = dim / numSubspaces;
    int remainder = dim % numSubspaces;
    int offset = 0;

    for (int m = 0; m < numSubspaces; m++) {
      int size = baseSize + (m < remainder ? 1 : 0);
      result[m][0] = size;
      result[m][1] = offset;
      offset += size;
    }

    return result;
  }

  /**
   * Samples training data using {@link ReservoirSampler} (deterministic seed 42). Returns up to
   * {@link #MAX_PQ_TRAINING_SET_SIZE} centered (or raw) vectors for k-means training.
   */
  private static float[][] sampleTrainingData(VectorDataset dataset, float[] centroid) {
    int n = dataset.size();
    int[] indices = ReservoirSampler.sampleIndices(n, MAX_PQ_TRAINING_SET_SIZE, new Random(42L));
    float[][] samples = new float[indices.length][];
    for (int i = 0; i < indices.length; i++) {
      samples[i] = maybeCentered(dataset.getVector(indices[i]), centroid);
    }
    return samples;
  }

  /**
   * Returns the vector centered by subtracting {@code centroid}, or the original vector (no copy)
   * when centering is disabled.
   */
  private static float[] maybeCentered(float[] vector, float[] centroid) {
    if (centroid == null) {
      return vector; // read-only use in k-means; no copy needed
    }
    float[] copy = Arrays.copyOf(vector, vector.length);
    VectorUtil.subInPlace(copy, centroid);
    return copy;
  }

  /**
   * Extracts sub-vectors for a given subspace from training data into a flat array suitable for
   * k-means.
   */
  private static float[] extractSubvectors(float[][] data, int offset, int subDim) {
    float[] result = new float[data.length * subDim];
    for (int i = 0; i < data.length; i++) {
      System.arraycopy(data[i], offset, result, i * subDim, subDim);
    }
    return result;
  }
}
