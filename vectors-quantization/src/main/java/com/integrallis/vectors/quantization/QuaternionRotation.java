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

import java.util.Random;

/**
 * Quaternion 4D block rotation ({@link Rotation} implementation). Groups dimensions into blocks of
 * 4 and applies an independent unit-quaternion rotation to each block, treating each 4D block as a
 * quaternion.
 *
 * <p>This implements the IsoQuant rotation strategy: SO(4) decomposes as SU(2) × SU(2), so a random
 * rotation in 4D is parameterized by two unit quaternions q<sub>L</sub> and q<sub>R</sub>. The
 * rotation is: {@code result = q_L * v * conj(q_R)} (Hamilton product sandwich).
 *
 * <p>Computational cost: <b>32 FMAs per 4D block</b> (two Hamilton products), or 8d FMAs total —
 * compared to d² for {@link RandomRotation} and 2d for {@link GivensRotation}. Despite being
 * slightly more expensive than Givens, quaternion rotations achieve better decorrelation because
 * each 4D rotation has 6 degrees of freedom (vs 1 for Givens 2D).
 *
 * <p>If the dimension is not a multiple of 4, trailing dimensions are left unrotated.
 *
 * @see Rotation
 * @see RandomRotation
 * @see GivensRotation
 */
public final class QuaternionRotation implements Rotation {

  // Per-block quaternions: [numBlocks][4] for q_L and q_R
  private final float[][] qL; // left quaternions
  private final float[][] qR; // right quaternions
  private final int dimension;
  private final int numBlocks;

  private QuaternionRotation(float[][] qL, float[][] qR, int dimension) {
    this.qL = qL;
    this.qR = qR;
    this.dimension = dimension;
    this.numBlocks = dimension / 4;
  }

  /**
   * Generates a quaternion rotation with random unit quaternion pairs for each 4D block.
   *
   * @param dimension the vector dimension
   * @param seed random seed for reproducibility
   * @return a quaternion rotation with d/4 random quaternion pairs
   */
  public static QuaternionRotation generate(int dimension, long seed) {
    if (dimension < 4) {
      throw new IllegalArgumentException(
          "QuaternionRotation requires dimension >= 4, got "
              + dimension
              + ". Use GivensRotation for dimension >= 2, or pad the vector to at least 4 dimensions.");
    }
    int numBlocks = dimension / 4;
    float[][] qL = new float[numBlocks][4];
    float[][] qR = new float[numBlocks][4];
    Random rng = new Random(seed);

    for (int i = 0; i < numBlocks; i++) {
      qL[i] = randomUnitQuaternion(rng);
      qR[i] = randomUnitQuaternion(rng);
    }

    return new QuaternionRotation(qL, qR, dimension);
  }

  @Override
  public float[] rotate(float[] vector) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + " for rotate(), got " + vector.length);
    }
    float[] result = new float[dimension];
    for (int b = 0; b < numBlocks; b++) {
      int base = b * 4;
      // Treat vector[base..base+3] as quaternion v = [w, x, y, z]
      float vw = vector[base];
      float vx = vector[base + 1];
      float vy = vector[base + 2];
      float vz = vector[base + 3];

      // temp = q_L * v (Hamilton product)
      float[] left = qL[b];
      float tw = left[0] * vw - left[1] * vx - left[2] * vy - left[3] * vz;
      float tx = left[0] * vx + left[1] * vw + left[2] * vz - left[3] * vy;
      float ty = left[0] * vy - left[1] * vz + left[2] * vw + left[3] * vx;
      float tz = left[0] * vz + left[1] * vy - left[2] * vx + left[3] * vw;

      // result = temp * conj(q_R) (Hamilton product with conjugate)
      float[] right = qR[b];
      // conj(q_R) = [w, -x, -y, -z]
      float rw = right[0];
      float rx = -right[1];
      float ry = -right[2];
      float rz = -right[3];

      result[base] = tw * rw - tx * rx - ty * ry - tz * rz;
      result[base + 1] = tw * rx + tx * rw + ty * rz - tz * ry;
      result[base + 2] = tw * ry - tx * rz + ty * rw + tz * rx;
      result[base + 3] = tw * rz + tx * ry - ty * rx + tz * rw;
    }
    // Trailing dimensions (dimension % 4 != 0): pass through unrotated
    for (int d = numBlocks * 4; d < dimension; d++) {
      result[d] = vector[d];
    }
    return result;
  }

  @Override
  public float[] inverseRotate(float[] vector) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + " for inverseRotate(), got " + vector.length);
    }
    float[] result = new float[dimension];
    for (int b = 0; b < numBlocks; b++) {
      int base = b * 4;
      float vw = vector[base];
      float vx = vector[base + 1];
      float vy = vector[base + 2];
      float vz = vector[base + 3];

      // Inverse: result = conj(q_L) * v * q_R
      float[] left = qL[b];
      // conj(q_L) = [w, -x, -y, -z]
      float lw = left[0];
      float lx = -left[1];
      float ly = -left[2];
      float lz = -left[3];

      // temp = conj(q_L) * v
      float tw = lw * vw - lx * vx - ly * vy - lz * vz;
      float tx = lw * vx + lx * vw + ly * vz - lz * vy;
      float ty = lw * vy - lx * vz + ly * vw + lz * vx;
      float tz = lw * vz + lx * vy - ly * vx + lz * vw;

      // result = temp * q_R
      float[] right = qR[b];
      result[base] = tw * right[0] - tx * right[1] - ty * right[2] - tz * right[3];
      result[base + 1] = tw * right[1] + tx * right[0] + ty * right[3] - tz * right[2];
      result[base + 2] = tw * right[2] - tx * right[3] + ty * right[0] + tz * right[1];
      result[base + 3] = tw * right[3] + tx * right[2] - ty * right[1] + tz * right[0];
    }
    for (int d = numBlocks * 4; d < dimension; d++) {
      result[d] = vector[d];
    }
    return result;
  }

  /**
   * Reconstructs a {@code QuaternionRotation} from previously serialized quaternion pairs. Used by
   * deserialization codecs to restore a trained rotation without re-generating random quaternions.
   *
   * @param dimension the vector dimension (must be >= 4)
   * @param qL left quaternions, shape [numBlocks][4] where numBlocks = dimension/4
   * @param qR right quaternions, shape [numBlocks][4]
   * @return a reconstructed quaternion rotation
   * @throws IllegalArgumentException if {@code dimension < 4} or array shapes don't match
   */
  public static QuaternionRotation fromQuaternions(int dimension, float[][] qL, float[][] qR) {
    if (dimension < 4) {
      throw new IllegalArgumentException(
          "QuaternionRotation requires dimension >= 4, got " + dimension);
    }
    int expectedBlocks = dimension / 4;
    if (qL.length != expectedBlocks || qR.length != expectedBlocks) {
      throw new IllegalArgumentException(
          "qL/qR arrays must have length " + expectedBlocks + " (dimension/4)");
    }
    return new QuaternionRotation(qL, qR, dimension);
  }

  /** Returns the left quaternions for each 4D block, shape [numBlocks][4]. */
  public float[][] qL() {
    return qL;
  }

  /** Returns the right quaternions for each 4D block, shape [numBlocks][4]. */
  public float[][] qR() {
    return qR;
  }

  @Override
  public int dimension() {
    return dimension;
  }

  /**
   * Generates a uniformly distributed random unit quaternion using Marsaglia's method: sample from
   * 4D standard normal, then normalize.
   */
  private static float[] randomUnitQuaternion(Random rng) {
    float w = (float) rng.nextGaussian();
    float x = (float) rng.nextGaussian();
    float y = (float) rng.nextGaussian();
    float z = (float) rng.nextGaussian();
    float norm = (float) Math.sqrt(w * w + x * x + y * y + z * z);
    if (norm == 0f) {
      return new float[] {1f, 0f, 0f, 0f}; // identity quaternion
    }
    float invNorm = 1f / norm;
    return new float[] {w * invNorm, x * invNorm, y * invNorm, z * invNorm};
  }
}
