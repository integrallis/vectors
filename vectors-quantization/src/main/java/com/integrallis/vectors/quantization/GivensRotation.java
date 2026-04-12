package com.integrallis.vectors.quantization;

import java.util.Random;

/**
 * Givens 2D pair rotation ({@link Rotation} implementation). Applies an independent random rotation
 * to each consecutive pair of dimensions, decorrelating coordinates with minimal overhead.
 *
 * <p>For a d-dimensional vector, the rotation groups dimensions into d/2 pairs and applies a 2D
 * Givens rotation to each pair:
 *
 * <pre>
 * [v0', v1'] = [cos(θ), -sin(θ)] [v0]
 *              [sin(θ),  cos(θ)] [v1]
 * </pre>
 *
 * <p>This is the fastest rotation strategy: <b>4 FMAs per 2D pair</b>, or 2d FMAs total — compared
 * to d² for {@link RandomRotation}. Despite its simplicity, it achieves competitive quantization
 * quality because per-coordinate decorrelation (not global mixing) is the primary driver of
 * quantization accuracy.
 *
 * <p>If the dimension is odd, the last dimension is left unrotated.
 *
 * @see Rotation
 * @see RandomRotation
 * @see QuaternionRotation
 */
public final class GivensRotation implements Rotation {

  private final float[] cos; // cos[i] = cos(angle[i]) for pair i
  private final float[] sin; // sin[i] = sin(angle[i]) for pair i
  private final int dimension;
  private final int numPairs;

  private GivensRotation(float[] cos, float[] sin, int dimension) {
    this.cos = cos;
    this.sin = sin;
    this.dimension = dimension;
    this.numPairs = dimension / 2;
  }

  /**
   * Generates a Givens rotation with random angles for each 2D pair.
   *
   * @param dimension the vector dimension
   * @param seed random seed for reproducibility
   * @return a Givens rotation with d/2 random angles
   */
  public static GivensRotation generate(int dimension, long seed) {
    if (dimension < 2) {
      throw new IllegalArgumentException(
          "GivensRotation requires dimension >= 2, got "
              + dimension
              + ". Use a higher-dimensional padded vector or a different rotation strategy.");
    }
    int numPairs = dimension / 2;
    float[] cos = new float[numPairs];
    float[] sin = new float[numPairs];
    Random rng = new Random(seed);

    for (int i = 0; i < numPairs; i++) {
      float angle = (float) (rng.nextDouble() * 2.0 * Math.PI);
      cos[i] = (float) Math.cos(angle);
      sin[i] = (float) Math.sin(angle);
    }

    return new GivensRotation(cos, sin, dimension);
  }

  @Override
  public float[] rotate(float[] vector) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Expected dimension " + dimension + " for rotate(), got " + vector.length);
    }
    float[] result = new float[dimension];
    for (int i = 0; i < numPairs; i++) {
      int d0 = i * 2;
      int d1 = d0 + 1;
      float c = cos[i];
      float s = sin[i];
      result[d0] = c * vector[d0] - s * vector[d1];
      result[d1] = s * vector[d0] + c * vector[d1];
    }
    // Odd dimension: last element passes through unrotated
    if (dimension % 2 != 0) {
      result[dimension - 1] = vector[dimension - 1];
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
    for (int i = 0; i < numPairs; i++) {
      int d0 = i * 2;
      int d1 = d0 + 1;
      float c = cos[i];
      float s = sin[i];
      // Inverse = transpose: negate sin
      result[d0] = c * vector[d0] + s * vector[d1];
      result[d1] = -s * vector[d0] + c * vector[d1];
    }
    if (dimension % 2 != 0) {
      result[dimension - 1] = vector[dimension - 1];
    }
    return result;
  }

  /**
   * Reconstructs a {@code GivensRotation} from previously serialized cos/sin arrays. Used by
   * deserialization codecs to restore a trained rotation without re-generating random angles.
   *
   * @param dimension the vector dimension (must be >= 2)
   * @param cos cosine values for each 2D pair (length must be dimension/2)
   * @param sin sine values for each 2D pair (length must be dimension/2)
   * @return a reconstructed Givens rotation
   * @throws IllegalArgumentException if dimension < 2 or array lengths don't match
   */
  public static GivensRotation fromCosSin(int dimension, float[] cos, float[] sin) {
    if (dimension < 2) {
      throw new IllegalArgumentException(
          "GivensRotation requires dimension >= 2, got " + dimension);
    }
    int expectedPairs = dimension / 2;
    if (cos.length != expectedPairs || sin.length != expectedPairs) {
      throw new IllegalArgumentException(
          "cos/sin arrays must have length " + expectedPairs + " (dimension/2)");
    }
    return new GivensRotation(cos, sin, dimension);
  }

  /** Returns the cosine values for each 2D pair. */
  public float[] cos() {
    return cos;
  }

  /** Returns the sine values for each 2D pair. */
  public float[] sin() {
    return sin;
  }

  @Override
  public int dimension() {
    return dimension;
  }
}
