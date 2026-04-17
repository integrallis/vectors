package com.integrallis.vectors.quantization;

import java.util.Random;

/**
 * Optimized Product Quantization (OPQ) — jointly learns an orthogonal rotation matrix R and a PQ
 * codebook to minimize vector reconstruction error.
 *
 * <p>Standard PQ assigns dimensions to subspaces in sequential order, which works poorly when
 * adjacent dimensions are correlated. OPQ finds a rotation that distributes energy evenly across
 * subspaces, significantly improving recall at the same compression ratio.
 *
 * <p>Algorithm (Ge et al., "Optimized Product Quantization", CVPR 2013):
 *
 * <ol>
 *   <li>Initialize R = random orthogonal matrix.
 *   <li>Rotate training data: Z = R · x for each x.
 *   <li>Train standard PQ on Z to obtain codebooks.
 *   <li>Compute reconstruction error: Ẑ = decode(encode(Z)).
 *   <li>Solve Procrustes problem: R ← argmin‖R·X − Ẑ‖₂ s.t. Rᵀ R = I. Implemented via
 *       Newton-iteration polar decomposition of Ẑᵀ·X.
 *   <li>Repeat from step 2 until convergence (typically 5–10 iterations).
 * </ol>
 *
 * <p>Computational complexity: O(n · d²) for the cross-correlation matrix + O(d³) for the polar
 * decomposition. Suitable for d ≤ 512 in most deployment scenarios; for d &gt; 512, training is
 * still correct but may take 10–60 s per iteration depending on hardware.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * VectorDataset data = new ArrayVectorDataset(vectors);
 * OptimizedProductQuantizer opq = OptimizedProductQuantizer.train(data, 16, 256, 10);
 * PQVectors compressed = opq.encodeAll(data);
 * float[][] adcTable = opq.buildADCTable(query, false);
 * }</pre>
 */
public final class OptimizedProductQuantizer implements Quantizer<PQVectors> {

  /** Maximum OPQ rotation-training samples (separate from PQ training budget). */
  static final int MAX_OPQ_ROTATION_SAMPLES = 10_000;

  /** Newton polar-decomposition convergence threshold. */
  private static final double POLAR_EPS = 1e-10;

  /** Maximum Newton iterations for polar decomposition. */
  private static final int POLAR_MAX_ITER = 30;

  private final ProductQuantizer pq;

  /** Row-major d×d rotation matrix: y = R * x means y[i] = sum_j R[i][j] * x[j]. */
  private final float[][] rotation;

  private final int dimension;

  private OptimizedProductQuantizer(ProductQuantizer pq, float[][] rotation) {
    this.pq = pq;
    this.rotation = rotation;
    this.dimension = pq.dimension();
  }

  // ---------------------------------------------------------------------------
  // Training
  // ---------------------------------------------------------------------------

  /**
   * Trains OPQ with default parameters (5 iterations, no global centering for PQ).
   *
   * @param dataset training vectors
   * @param numSubspaces PQ sub-vector count (M)
   * @param numClusters centroids per subspace (Ks, must be ≤ 256)
   * @return trained OPQ
   */
  public static OptimizedProductQuantizer train(
      VectorDataset dataset, int numSubspaces, int numClusters) {
    return train(dataset, numSubspaces, numClusters, 5, 42L);
  }

  /**
   * Trains OPQ with full parameter control.
   *
   * @param dataset training vectors
   * @param numSubspaces PQ sub-vector count (M)
   * @param numClusters centroids per subspace (Ks, must be ≤ 256)
   * @param numIterations alternating-optimization iterations (5–20 recommended)
   * @param seed random seed for initial rotation
   * @return trained OPQ
   */
  public static OptimizedProductQuantizer train(
      VectorDataset dataset, int numSubspaces, int numClusters, int numIterations, long seed) {
    if (numIterations < 1) throw new IllegalArgumentException("numIterations must be >= 1");

    int n = dataset.size();
    int d = dataset.dimension();

    // Sample training vectors for rotation learning (up to MAX_OPQ_ROTATION_SAMPLES)
    Random rng = new Random(seed);
    int[] sampleIdx = ReservoirSampler.sampleIndices(n, MAX_OPQ_ROTATION_SAMPLES, rng);
    int ns = sampleIdx.length;
    float[][] X = new float[ns][d];
    for (int i = 0; i < ns; i++) X[i] = dataset.getVector(sampleIdx[i]).clone();

    // Initialize rotation to a random orthogonal matrix (Gram-Schmidt on random matrix)
    float[][] R = randomOrthogonal(d, rng);

    ProductQuantizer pq = null;

    for (int iter = 0; iter < numIterations; iter++) {
      // Step 1: rotate training sample, train PQ on rotated data
      float[][] Z = applyRotation(X, R);
      pq = ProductQuantizer.train(new ArrayVectorDataset(Z), numSubspaces, numClusters, false);

      // Step 2: compute PQ reconstructions Ẑ for each sampled vector
      float[][] Zhat = new float[ns][d];
      for (int i = 0; i < ns; i++) Zhat[i] = pq.decode(pq.encode(Z[i]));

      // Step 3: solve Procrustes — R = argmin‖R·X − Ẑ‖ s.t. Rᵀ R = I
      //   Optimal R = V Uᵀ from SVD(Ẑᵀ · X) = U Σ Vᵀ
      //   Equivalent to: R = polar(Ẑᵀ · X)ᵀ
      float[][] M = crossCorrelation(Zhat, X); // d×d: Ẑᵀ · X
      float[][] P = polarDecomposition(M);
      R = transpose(P);
    }

    // Final PQ training on the full dataset (rotated), using standard train()
    // (which applies its own reservoir-sampling budget)
    float[][] rotation = R;
    VectorDataset rotated =
        new VectorDataset() {
          @Override
          public int size() {
            return dataset.size();
          }

          @Override
          public int dimension() {
            return d;
          }

          @Override
          public float[] getVector(int ordinal) {
            return rotateVector(dataset.getVector(ordinal), rotation);
          }
        };
    pq = ProductQuantizer.train(rotated, numSubspaces, numClusters, false);

    return new OptimizedProductQuantizer(pq, rotation);
  }

  // ---------------------------------------------------------------------------
  // Quantizer interface
  // ---------------------------------------------------------------------------

  @Override
  public byte[] encode(float[] vector) {
    return pq.encode(rotateVector(vector, rotation));
  }

  @Override
  public float encode(float[] vector, byte[] dst) {
    return pq.encode(rotateVector(vector, rotation), dst);
  }

  @Override
  public float[] decode(byte[] encoded) {
    // Decoded vector is in rotated space; apply Rᵀ to map back to original space.
    return applyTransposeRotation(pq.decode(encoded), rotation);
  }

  @Override
  public PQVectors encodeAll(VectorDataset dataset) {
    float[][] rotation = this.rotation;
    VectorDataset rotated =
        new VectorDataset() {
          @Override
          public int size() {
            return dataset.size();
          }

          @Override
          public int dimension() {
            return dataset.dimension();
          }

          @Override
          public float[] getVector(int ordinal) {
            return rotateVector(dataset.getVector(ordinal), rotation);
          }
        };
    return pq.encodeAll(rotated);
  }

  @Override
  public float compressionRatio() {
    return pq.compressionRatio();
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // ---------------------------------------------------------------------------
  // OPQ-specific API
  // ---------------------------------------------------------------------------

  /**
   * Builds an ADC lookup table for the given query. The query is rotated before building the table
   * to match the encoded (rotated) database vectors.
   *
   * @param query the raw (unrotated) query vector
   * @param useDotProduct true for inner-product distances, false for L2
   * @return M × Ks lookup table (same layout as {@link ProductQuantizer#buildADCTable})
   */
  public float[][] buildADCTable(float[] query, boolean useDotProduct) {
    return pq.buildADCTable(rotateVector(query, rotation), useDotProduct);
  }

  /** Returns the learned d×d rotation matrix (row-major). */
  public float[][] rotation() {
    return rotation;
  }

  /** Returns the underlying {@link ProductQuantizer} (trained on rotated data). */
  public ProductQuantizer productQuantizer() {
    return pq;
  }

  // ---------------------------------------------------------------------------
  // Package-private matrix helpers (accessed by tests)
  // ---------------------------------------------------------------------------

  /** Applies row-vector rotation: result[i] = R · X[i] (row-major R). */
  static float[][] applyRotation(float[][] X, float[][] R) {
    int n = X.length;
    int d = R.length;
    float[][] result = new float[n][d];
    for (int i = 0; i < n; i++) result[i] = rotateVector(X[i], R);
    return result;
  }

  /** y = R · x: y[i] = sum_j R[i][j] * x[j]. */
  static float[] rotateVector(float[] x, float[][] R) {
    int d = x.length;
    float[] y = new float[d];
    for (int i = 0; i < d; i++) {
      float s = 0f;
      for (int j = 0; j < d; j++) s += R[i][j] * x[j];
      y[i] = s;
    }
    return y;
  }

  /** y = Rᵀ · x: y[i] = sum_j R[j][i] * x[j] (inverse rotation). */
  private static float[] applyTransposeRotation(float[] x, float[][] R) {
    int d = x.length;
    float[] y = new float[d];
    for (int i = 0; i < d; i++) {
      float s = 0f;
      for (int j = 0; j < d; j++) s += R[j][i] * x[j]; // Rᵀ[i][j] = R[j][i]
      y[i] = s;
    }
    return y;
  }

  /** M = A^T · B (d×d cross-correlation matrix, A and B are n×d). */
  static float[][] crossCorrelation(float[][] A, float[][] B) {
    int n = A.length;
    int d = A[0].length;
    float[][] M = new float[d][d];
    for (int k = 0; k < n; k++) {
      float[] ak = A[k];
      float[] bk = B[k];
      for (int i = 0; i < d; i++) {
        float ai = ak[i];
        for (int j = 0; j < d; j++) M[i][j] += ai * bk[j];
      }
    }
    return M;
  }

  /**
   * Polar decomposition via Newton iteration: converges to the orthogonal factor P of M = P·S.
   *
   * <p>Each step: P ← ½(P + (Pᵀ)⁻¹). Converges quadratically.
   */
  static float[][] polarDecomposition(float[][] M) {
    int d = M.length;
    float[][] P = deepCopy(M);
    for (int iter = 0; iter < POLAR_MAX_ITER; iter++) {
      float[][] Pt = transpose(P);
      float[][] PtInv = invert(Pt);
      double diff = 0.0;
      float[][] Pnew = new float[d][d];
      for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
          Pnew[i][j] = 0.5f * (P[i][j] + PtInv[i][j]);
          double delta = Pnew[i][j] - P[i][j];
          diff += delta * delta;
        }
      }
      P = Pnew;
      if (diff < POLAR_EPS) break;
    }
    return P;
  }

  /** Gauss-Jordan elimination with partial pivoting; returns the inverse of A (d×d). */
  static float[][] invert(float[][] A) {
    int d = A.length;
    float[][] U = deepCopy(A);
    float[][] Inv = new float[d][d];

    // Initialize Inv as identity
    for (int i = 0; i < d; i++) Inv[i][i] = 1f;

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < d; col++) {
      // Find pivot row
      int maxRow = col;
      float maxVal = Math.abs(U[col][col]);
      for (int row = col + 1; row < d; row++) {
        float absVal = Math.abs(U[row][col]);
        if (absVal > maxVal) {
          maxVal = absVal;
          maxRow = row;
        }
      }
      // Swap rows in U and Inv
      float[] tmpU = U[col];
      U[col] = U[maxRow];
      U[maxRow] = tmpU;
      float[] tmpI = Inv[col];
      Inv[col] = Inv[maxRow];
      Inv[maxRow] = tmpI;

      float diag = U[col][col];
      if (Math.abs(diag) < 1e-12f) continue; // singular or near-singular
      for (int row = col + 1; row < d; row++) {
        float factor = U[row][col] / diag;
        for (int k = col; k < d; k++) U[row][k] -= factor * U[col][k];
        for (int k = 0; k < d; k++) Inv[row][k] -= factor * Inv[col][k];
      }
    }
    // Back substitution
    for (int col = d - 1; col >= 0; col--) {
      float diag = U[col][col];
      if (Math.abs(diag) < 1e-12f) continue;
      for (int k = 0; k < d; k++) Inv[col][k] /= diag;
      for (int row = 0; row < col; row++) {
        float factor = U[row][col];
        for (int k = 0; k < d; k++) Inv[row][k] -= factor * Inv[col][k];
      }
    }
    return Inv;
  }

  /** Random orthogonal matrix via Modified Gram-Schmidt QR of a random matrix. */
  static float[][] randomOrthogonal(int d, Random rng) {
    // Fill with Gaussian random entries
    float[][] Q = new float[d][d];
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) Q[i][j] = (float) rng.nextGaussian();
    // Gram-Schmidt orthonormalization (column-wise)
    for (int i = 0; i < d; i++) {
      // Subtract projections onto earlier columns
      for (int j = 0; j < i; j++) {
        float dot = 0f;
        for (int k = 0; k < d; k++) dot += Q[k][i] * Q[k][j];
        for (int k = 0; k < d; k++) Q[k][i] -= dot * Q[k][j];
      }
      // Normalize column i
      float norm = 0f;
      for (int k = 0; k < d; k++) norm += Q[k][i] * Q[k][i];
      norm = (float) Math.sqrt(norm);
      if (norm > 1e-12f) for (int k = 0; k < d; k++) Q[k][i] /= norm;
    }
    // Return as row-major (transpose Q so R[i][j] = column-j of Q[i])
    float[][] R = new float[d][d];
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) R[i][j] = Q[j][i];
    return R;
  }

  static float[][] transpose(float[][] M) {
    int d = M.length;
    float[][] T = new float[d][d];
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) T[i][j] = M[j][i];
    return T;
  }

  static float[][] deepCopy(float[][] M) {
    float[][] C = new float[M.length][];
    for (int i = 0; i < M.length; i++) C[i] = M[i].clone();
    return C;
  }
}
