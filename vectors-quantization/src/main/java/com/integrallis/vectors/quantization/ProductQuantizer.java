package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Random;

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

  private ProductQuantizer(
      int dimension,
      int numSubspaces,
      int numClusters,
      int[][] subspaceSizesAndOffsets,
      float[][] codebooks,
      float[] globalCentroid) {
    this.dimension = dimension;
    this.numSubspaces = numSubspaces;
    this.numClusters = numClusters;
    this.subspaceSizesAndOffsets = subspaceSizesAndOffsets;
    this.codebooks = codebooks;
    this.globalCentroid = globalCentroid;
  }

  // --- Factory methods ---

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
   * Trains a PQ with full control over parameters.
   *
   * @param dataset the training data
   * @param numSubspaces number of sub-vector partitions (M)
   * @param numClusters centroids per subspace (Ks), must be &le; 256
   * @param center if true, subtract global centroid before quantization
   * @return a trained product quantizer
   */
  public static ProductQuantizer train(
      VectorDataset dataset, int numSubspaces, int numClusters, boolean center) {
    int dim = dataset.dimension();
    validateParameters(dim, numSubspaces, numClusters);

    int[][] sizesAndOffsets = computeSubspaceSizesAndOffsets(dim, numSubspaces);

    // Optionally compute and subtract global centroid
    float[] centroid = center ? dataset.computeCentroid() : null;

    // Sample training data if dataset is large
    float[][] trainingData = sampleTrainingData(dataset, centroid);

    // Train codebook for each subspace via k-means++
    Random rng = new Random(42L);
    float[][] codebooks = new float[numSubspaces][];

    for (int m = 0; m < numSubspaces; m++) {
      int subDim = sizesAndOffsets[m][0];
      int offset = sizesAndOffsets[m][1];

      // Extract sub-vectors for this subspace
      float[] subVectors = extractSubvectors(trainingData, offset, subDim);

      // Cluster
      codebooks[m] =
          KMeansPlusPlusClusterer.cluster(
              subVectors,
              trainingData.length,
              subDim,
              numClusters,
              KMeansPlusPlusClusterer.DEFAULT_MAX_ITERATIONS,
              rng);
    }

    return new ProductQuantizer(
        dim, numSubspaces, numClusters, sizesAndOffsets, codebooks, centroid);
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
   * Returns the codebook for subspace m. The codebook is a flat array of size {@code numClusters *
   * subDim}, where entry c starts at index {@code c * subDim}.
   */
  float[] codebook(int m) {
    return codebooks[m];
  }

  /** Returns the subspace size and offset as [size, offset]. */
  int[] subspaceSizeAndOffset(int m) {
    return subspaceSizesAndOffsets[m];
  }

  /** Returns the global centroid, or null if centering is not enabled. */
  float[] globalCentroid() {
    return globalCentroid;
  }

  // --- ADC lookup table computation ---

  /**
   * Builds the ADC (Asymmetric Distance Computation) lookup table for a query. Returns a 2D table
   * where {@code table[m][k]} is the partial distance/similarity between query sub-vector m and
   * centroid k.
   *
   * @param query the query vector (full precision, before centering)
   * @param useDotProduct true to fill with dot products, false for squared L2 distances
   * @return lookup table of size [numSubspaces][numClusters]
   */
  float[][] buildADCTable(float[] query, boolean useDotProduct) {
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
