package com.integrallis.vectors.core;

/**
 * Mathematical utility methods for vector operations. Provides FMA (fused multiply-add) helpers and
 * common numerical operations used throughout the library.
 */
public final class MathUtil {

  private MathUtil() {}

  /** Threshold for considering a vector as unit-length (squared norm within this of 1.0). */
  private static final float UNIT_VECTOR_EPSILON = 1e-5f;

  /**
   * Returns true if the given squared norm indicates a unit vector. A squared norm is considered
   * unit if {@code |squaredNorm - 1.0| < epsilon}.
   */
  public static boolean isUnitVector(float squaredNorm) {
    return Math.abs(squaredNorm - 1.0f) < UNIT_VECTOR_EPSILON;
  }

  /** Returns the square of a float value. */
  public static float square(float x) {
    return x * x;
  }

  /**
   * Scalar fused multiply-add: returns {@code a * b + c}. Uses {@link Math#fma} when the platform
   * has fast scalar FMA, otherwise falls back to {@code a * b + c}.
   */
  public static float fma(float a, float b, float c) {
    if (PanamaConstants.HAS_FAST_SCALAR_FMA) {
      return Math.fma(a, b, c);
    } else {
      return a * b + c;
    }
  }
}
