package com.integrallis.vectors.quantization;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Lloyd-Max optimal scalar quantizer for the coordinate distribution of randomly rotated unit
 * vectors. Precomputes MSE-optimal quantization levels (centroids) and decision boundaries for a
 * given (dimension, bits) pair.
 *
 * <p><b>Theoretical foundation:</b> After applying a random orthogonal rotation to a d-dimensional
 * unit vector, each coordinate follows a Beta((d-1)/2, 1/2) distribution on [-1, 1]. For practical
 * dimensions (d ≥ 64), this is well-approximated by N(0, 1/d). The Lloyd-Max algorithm finds the
 * MSE-optimal scalar quantizer for this distribution.
 *
 * <p><b>Algorithm:</b> Continuous 1D k-means (Lloyd-Max iteration):
 *
 * <ol>
 *   <li>Initialize 2<sup>bits</sup> centroids uniformly in [-3.5σ, +3.5σ] where σ = 1/√d
 *   <li>Compute boundaries as midpoints between adjacent centroids
 *   <li>Update each centroid as the conditional expectation E[X | X ∈ partition_i]
 *   <li>Repeat until convergence (max centroid shift &lt; tolerance)
 * </ol>
 *
 * <p>The codebook is precomputed once and reused for all vectors. It is shared across all
 * rotation-based quantizers (TurboQuant, Extended RaBitQ).
 *
 * <p>Usage:
 *
 * <pre>{@code
 * LloydMaxCodebook codebook = LloydMaxCodebook.compute(128, 4); // 4-bit, dim=128
 * int index = codebook.quantize(0.05f);     // find nearest centroid
 * float value = codebook.dequantize(index);  // look up centroid value
 * }</pre>
 */
public final class LloydMaxCodebook {

  /**
   * Cache of computed codebooks keyed by {@code (dimension << 32) | bits}. Computing an 8-bit
   * codebook (256 levels) requires ~1 billion {@code exp()} evaluations; caching avoids repeating
   * this cost on repeated {@link #compute} calls with the same (dimension, bits) pair. The map is
   * lazily populated and safe for concurrent access via {@code computeIfAbsent}.
   */
  private static final ConcurrentHashMap<Long, LloydMaxCodebook> CACHE = new ConcurrentHashMap<>();

  /** Default convergence tolerance for Lloyd-Max iteration. */
  private static final double TOLERANCE = 1e-10;

  /** Maximum iterations for Lloyd-Max convergence. */
  private static final int MAX_ITERATIONS = 200;

  /** Number of Simpson's rule intervals for numerical integration. */
  private static final int INTEGRATION_INTERVALS = 10_000;

  /** Initialization range in sigma units. Centroids are spaced in [-k*σ, +k*σ]. */
  private static final double INIT_SIGMA_RANGE = 3.5;

  /** Integration truncation range in sigma units for conditional expectation computation. */
  private static final double INTEGRATION_SIGMA_RANGE = 6.0;

  private final float[] centroids; // sorted quantization levels, length = 2^bits
  private final float[] boundaries; // sorted decision boundaries, length = 2^bits - 1
  private final int bits;
  private final int numLevels;

  private LloydMaxCodebook(float[] centroids, float[] boundaries, int bits) {
    this.centroids = centroids;
    this.boundaries = boundaries;
    this.bits = bits;
    this.numLevels = centroids.length;
  }

  /**
   * Computes the Lloyd-Max optimal codebook for the given dimension and bit-width.
   *
   * <p>Uses the Gaussian approximation N(0, 1/d) for the coordinate distribution of randomly
   * rotated unit vectors. This approximation is accurate for d ≥ 64.
   *
   * @param dimension the vector dimension (determines the variance σ² = 1/d)
   * @param bits the number of quantization bits (1-8)
   * @return the precomputed codebook
   * @throws IllegalArgumentException if bits is not in [1, 8] or dimension &lt; 1
   */
  public static LloydMaxCodebook compute(int dimension, int bits) {
    if (bits < 1 || bits > 8) {
      throw new IllegalArgumentException("Bits must be in [1, 8], got " + bits);
    }
    if (dimension < 1) {
      throw new IllegalArgumentException("Dimension must be >= 1, got " + dimension);
    }

    // Cache result: computing an 8-bit codebook performs ~1B exp() calls; identical (dim, bits)
    // always produces identical output, so we cache by (dimension << 32 | bits).
    long cacheKey = ((long) dimension << 32) | (bits & 0xFFFFFFFFL);
    return CACHE.computeIfAbsent(cacheKey, k -> computeUncached(dimension, bits));
  }

  /** Performs the actual Lloyd-Max computation (called at most once per (dimension, bits) pair). */
  private static LloydMaxCodebook computeUncached(int dimension, int bits) {
    int numLevels = 1 << bits;
    double sigma = 1.0 / Math.sqrt(dimension);

    // Special case: 1-bit is just the sign function
    if (bits == 1) {
      // Optimal 1-bit quantizer for symmetric distribution: boundary at 0,
      // centroids at ± E[|X|] = ± σ * sqrt(2/π)
      float centroidMag = (float) (sigma * Math.sqrt(2.0 / Math.PI));
      return new LloydMaxCodebook(new float[] {-centroidMag, centroidMag}, new float[] {0f}, 1);
    }

    // Initialize centroids uniformly in [-INIT_SIGMA_RANGE*σ, +INIT_SIGMA_RANGE*σ]
    double lo = -INIT_SIGMA_RANGE * sigma;
    double hi = INIT_SIGMA_RANGE * sigma;
    double[] cents = new double[numLevels];
    for (int i = 0; i < numLevels; i++) {
      cents[i] = lo + (hi - lo) * (i + 0.5) / numLevels;
    }

    double[] bounds = new double[numLevels - 1];

    // Lloyd-Max iteration
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
      // Step 1: boundaries = midpoints between adjacent centroids
      for (int i = 0; i < bounds.length; i++) {
        bounds[i] = (cents[i] + cents[i + 1]) / 2.0;
      }

      // Step 2: update centroids as conditional expectations
      double maxShift = 0;
      for (int i = 0; i < numLevels; i++) {
        double a = (i == 0) ? -INTEGRATION_SIGMA_RANGE * sigma : bounds[i - 1];
        double b = (i == numLevels - 1) ? INTEGRATION_SIGMA_RANGE * sigma : bounds[i];

        double numerator = integrateXTimesGaussian(a, b, sigma);
        double denominator = integrateGaussian(a, b, sigma);

        double newCentroid = denominator > 0 ? numerator / denominator : (a + b) / 2.0;
        maxShift = Math.max(maxShift, Math.abs(newCentroid - cents[i]));
        cents[i] = newCentroid;
      }

      if (maxShift < TOLERANCE) {
        break;
      }
    }

    // Final boundaries after convergence
    for (int i = 0; i < bounds.length; i++) {
      bounds[i] = (cents[i] + cents[i + 1]) / 2.0;
    }

    // Convert to float
    float[] centroidsF = new float[numLevels];
    float[] boundariesF = new float[numLevels - 1];
    for (int i = 0; i < numLevels; i++) {
      centroidsF[i] = (float) cents[i];
    }
    for (int i = 0; i < bounds.length; i++) {
      boundariesF[i] = (float) bounds[i];
    }

    return new LloydMaxCodebook(centroidsF, boundariesF, bits);
  }

  /**
   * Quantizes a scalar value to the nearest centroid index.
   *
   * @param value the scalar value to quantize
   * @return the index of the nearest centroid (0 to 2^bits - 1)
   */
  public int quantize(float value) {
    // Binary search on boundaries
    int idx = Arrays.binarySearch(boundaries, value);
    if (idx >= 0) {
      // Exact match on boundary — assign to right partition
      return idx + 1;
    }
    // binarySearch returns -(insertion point) - 1
    return -(idx + 1);
  }

  /**
   * Dequantizes a centroid index back to the centroid value.
   *
   * @param index the centroid index (0 to 2^bits - 1)
   * @return the centroid value
   * @throws IndexOutOfBoundsException if index is out of range
   */
  public float dequantize(int index) {
    return centroids[index];
  }

  /** Returns the quantization bit-width. */
  public int bits() {
    return bits;
  }

  /** Returns the number of quantization levels (2^bits). */
  public int numLevels() {
    return numLevels;
  }

  /** Returns a copy of the centroids array. */
  public float[] centroids() {
    return centroids.clone();
  }

  /** Returns a copy of the boundaries array. */
  public float[] boundaries() {
    return boundaries.clone();
  }

  // --- Numerical integration helpers ---

  /**
   * Integrates x * p(x) over [a, b] where p(x) is N(0, σ²) using composite Simpson's rule. Used to
   * compute conditional expectations for Lloyd-Max centroid updates.
   */
  private static double integrateXTimesGaussian(double a, double b, double sigma) {
    if (a >= b) return 0;
    int n = INTEGRATION_INTERVALS;
    if (n % 2 != 0) n++;
    double h = (b - a) / n;
    double invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);

    double sum = xTimesGaussianUnnorm(a, invTwoSigmaSq) + xTimesGaussianUnnorm(b, invTwoSigmaSq);
    for (int i = 1; i < n; i++) {
      double x = a + i * h;
      double val = xTimesGaussianUnnorm(x, invTwoSigmaSq);
      sum += (i % 2 == 0) ? 2.0 * val : 4.0 * val;
    }
    double normalization = 1.0 / (sigma * Math.sqrt(2.0 * Math.PI));
    return sum * h / 3.0 * normalization;
  }

  /**
   * Integrates p(x) over [a, b] where p(x) is N(0, σ²) using composite Simpson's rule. Used to
   * compute partition probabilities for Lloyd-Max centroid updates.
   */
  private static double integrateGaussian(double a, double b, double sigma) {
    if (a >= b) return 0;
    int n = INTEGRATION_INTERVALS;
    if (n % 2 != 0) n++;
    double h = (b - a) / n;
    double invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);

    double sum = gaussianUnnorm(a, invTwoSigmaSq) + gaussianUnnorm(b, invTwoSigmaSq);
    for (int i = 1; i < n; i++) {
      double x = a + i * h;
      double val = gaussianUnnorm(x, invTwoSigmaSq);
      sum += (i % 2 == 0) ? 2.0 * val : 4.0 * val;
    }
    double normalization = 1.0 / (sigma * Math.sqrt(2.0 * Math.PI));
    return sum * h / 3.0 * normalization;
  }

  /** Unnormalized Gaussian: exp(-x² / (2σ²)). The normalization constant is applied once. */
  private static double gaussianUnnorm(double x, double invTwoSigmaSq) {
    return Math.exp(-x * x * invTwoSigmaSq);
  }

  /** x * unnormalized Gaussian. */
  private static double xTimesGaussianUnnorm(double x, double invTwoSigmaSq) {
    return x * Math.exp(-x * x * invTwoSigmaSq);
  }
}
