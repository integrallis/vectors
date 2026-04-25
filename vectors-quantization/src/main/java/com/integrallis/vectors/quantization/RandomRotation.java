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
import java.util.Random;

/**
 * Dense random orthogonal rotation matrix ({@link Rotation} implementation). Generates a D×D
 * orthogonal matrix via QR decomposition (Modified Gram-Schmidt) of a random Gaussian matrix. The
 * rotation maps vectors to a space where per-coordinate quantization produces better distance
 * estimators by decorrelating coordinate dependencies.
 *
 * <p>This is the densest (and most expensive) rotation strategy: O(d²) FMAs per rotation. For
 * faster alternatives with competitive or better quality, see {@link GivensRotation} (O(d) FMAs)
 * and {@link QuaternionRotation} (O(d) FMAs).
 *
 * <p>The rotation matrix Q satisfies Q<sup>T</sup>Q = I (orthogonal), so it preserves all pairwise
 * distances and inner products. Both the matrix and its transpose are stored for SIMD-accelerated
 * {@code rotate()} and {@code inverseRotate()} — each output element is computed as a SIMD dot
 * product of a contiguous matrix row with the input vector via {@link
 * VectorUtil#dotProduct(float[], float[])}.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * RandomRotation rot = RandomRotation.generate(128, 42L);
 * float[] rotated = rot.rotate(vector);
 * float[] recovered = rot.inverseRotate(rotated);  // ≈ vector
 * }</pre>
 *
 * @see Rotation
 * @see GivensRotation
 * @see QuaternionRotation
 */
public final class RandomRotation implements Rotation {

  private final float[][] matrix; // D×D orthogonal matrix (row-major)
  private final float[][] matrixT; // D×D transpose (row-major) for SIMD inverseRotate
  private final int dimension;

  private RandomRotation(float[][] matrix, float[][] matrixT, int dimension) {
    this.matrix = matrix;
    this.matrixT = matrixT;
    this.dimension = dimension;
  }

  /**
   * Generates a random orthogonal rotation matrix via QR decomposition of a Gaussian random matrix.
   *
   * <p>Uses Modified Gram-Schmidt for numerical stability. The resulting matrix is orthogonal
   * (Q<sup>T</sup>Q = I) and deterministic for a given seed.
   *
   * @param dimension the matrix dimension (should be padded to a multiple of 64)
   * @param seed random seed for reproducibility
   * @return a random orthogonal rotation
   */
  public static RandomRotation generate(int dimension, long seed) {
    if (dimension < 1) {
      throw new IllegalArgumentException(
          "RandomRotation requires dimension >= 1, got " + dimension);
    }
    Random rng = new Random(seed);
    float[][] q = new float[dimension][dimension];

    // Fill with Gaussian random values
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
        q[i][j] = (float) rng.nextGaussian();
      }
    }

    // Modified Gram-Schmidt QR decomposition (column-oriented)
    for (int col = 0; col < dimension; col++) {
      // Subtract projections onto all previous orthogonal columns
      for (int prev = 0; prev < col; prev++) {
        float dot = columnDotProduct(q, col, q, prev, dimension);
        for (int row = 0; row < dimension; row++) {
          q[row][col] -= dot * q[row][prev];
        }
      }

      // Normalize column
      float normSq = columnDotProduct(q, col, q, col, dimension);
      float norm = (float) Math.sqrt(normSq);
      if (norm > 0) {
        float invNorm = 1.0f / norm;
        for (int row = 0; row < dimension; row++) {
          q[row][col] *= invNorm;
        }
      }
    }

    // Precompute transpose for SIMD-accelerated inverseRotate
    float[][] qT = transpose(q, dimension);

    return new RandomRotation(q, qT, dimension);
  }

  /**
   * Applies the rotation to a vector: result[i] = dot(matrix[i], vector).
   *
   * <p>Each output element is a SIMD-accelerated dot product of a matrix row with the input vector
   * via {@link VectorUtil#dotProduct(float[], float[])}.
   *
   * @param vector the input vector (must have length == dimension)
   * @return the rotated vector
   */
  @Override
  public float[] rotate(float[] vector) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + " for rotate(), got " + vector.length);
    }
    float[] result = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      result[i] = VectorUtil.dotProduct(matrix[i], vector);
    }
    return result;
  }

  /**
   * Applies the inverse (transpose) rotation: result = Q<sup>T</sup> * vector.
   *
   * <p>Since Q is orthogonal, Q<sup>-1</sup> = Q<sup>T</sup>. Each output element is a
   * SIMD-accelerated dot product of a transpose row with the input vector.
   *
   * @param vector the input vector in rotated space
   * @return the vector in original space
   */
  @Override
  public float[] inverseRotate(float[] vector) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + " for inverseRotate(), got " + vector.length);
    }
    float[] result = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      result[i] = VectorUtil.dotProduct(matrixT[i], vector);
    }
    return result;
  }

  /**
   * Reconstructs a {@code RandomRotation} from a previously serialized matrix and its transpose.
   * Used by deserialization codecs to restore a trained rotation without re-running QR
   * decomposition.
   *
   * <p>The caller must provide both Q and Q<sup>T</sup> (not recomputed) to guarantee bit-identical
   * round-trip serialization — recomputing Q<sup>T</sup> from Q would accumulate float rounding
   * errors at d=64+.
   *
   * @param dimension the matrix dimension
   * @param q the D×D orthogonal matrix (row-major)
   * @param qt the D×D transpose of q (row-major)
   * @return a reconstructed rotation
   * @throws IllegalArgumentException if dimension < 1 or matrix shapes don't match
   */
  public static RandomRotation fromMatrix(int dimension, float[][] q, float[][] qt) {
    if (dimension < 1) {
      throw new IllegalArgumentException(
          "RandomRotation requires dimension >= 1, got " + dimension);
    }
    if (q.length != dimension || qt.length != dimension) {
      throw new IllegalArgumentException("Matrix row count must match dimension " + dimension);
    }
    return new RandomRotation(q, qt, dimension);
  }

  /** Returns the D×D orthogonal rotation matrix (row-major). */
  public float[][] matrix() {
    return matrix;
  }

  /** Returns the D×D transpose of the rotation matrix (row-major). */
  public float[][] matrixT() {
    return matrixT;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  // --- Internal helpers (training-time only, not on hot path) ---

  /** Dot product of two columns in a row-major matrix. Training-time only. */
  private static float columnDotProduct(
      float[][] a, int colA, float[][] b, int colB, int dimension) {
    float dot = 0f;
    for (int row = 0; row < dimension; row++) {
      dot += a[row][colA] * b[row][colB];
    }
    return dot;
  }

  /** Transposes a square matrix. Training-time only. */
  private static float[][] transpose(float[][] m, int dimension) {
    float[][] t = new float[dimension][dimension];
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j < dimension; j++) {
        t[j][i] = m[i][j];
      }
    }
    return t;
  }
}
