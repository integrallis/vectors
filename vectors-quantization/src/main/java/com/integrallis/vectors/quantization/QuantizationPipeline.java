package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Objects;

/**
 * Fluent entry point for the standalone quantization API.
 *
 * <p>A {@code QuantizationPipeline} bundles a trained {@link Quantizer} with its similarity
 * function so that the same object can be used for encoding, decoding, batch compression, and
 * approximate scoring without exposing the underlying quantizer family to the caller.
 *
 * <p><b>Typical usage:</b>
 *
 * <pre>{@code
 * VectorDataset corpus = new ArrayVectorDataset(vectors);
 *
 * // Scalar INT8 — one line
 * QuantizationPipeline sq = QuantizationPipeline.scalarInt8().train(corpus);
 *
 * // Product quantization with custom subspaces
 * QuantizationPipeline pq = QuantizationPipeline.product(8).train(corpus);
 *
 * // Full builder control
 * QuantizationPipeline opq = QuantizationPipeline.builder()
 *     .quantizerType(QuantizerType.OPTIMIZED_PRODUCT)
 *     .subspaces(16)
 *     .iterations(10)
 *     .similarity(SimilarityFunction.DOT_PRODUCT)
 *     .train(corpus);
 *
 * byte[]           code  = opq.encode(query);
 * CompressedVectors comp  = opq.compress(corpus);
 * ScoreFunction    scorer = opq.scorerFor(query, comp);
 * }</pre>
 */
public final class QuantizationPipeline {

  private final Quantizer<?> quantizer;
  private final QuantizerType type;
  private final SimilarityFunction similarity;

  private QuantizationPipeline(Quantizer<?> quantizer, QuantizerType type, SimilarityFunction sim) {
    this.quantizer = Objects.requireNonNull(quantizer);
    this.type = Objects.requireNonNull(type);
    this.similarity = Objects.requireNonNull(sim);
  }

  // ---------------------------------------------------------------------------
  // Preset factory methods
  // ---------------------------------------------------------------------------

  /** Returns a builder pre-configured for 8-bit scalar quantization. */
  public static Builder scalarInt8() {
    return new Builder().quantizerType(QuantizerType.SCALAR_INT8);
  }

  /** Returns a builder pre-configured for 4-bit scalar quantization. */
  public static Builder scalarInt4() {
    return new Builder().quantizerType(QuantizerType.SCALAR_INT4);
  }

  /** Returns a builder pre-configured for Product Quantization with {@code subspaces} subspaces. */
  public static Builder product(int subspaces) {
    return new Builder().quantizerType(QuantizerType.PRODUCT).subspaces(subspaces);
  }

  /** Returns a builder pre-configured for Optimized PQ with {@code subspaces} subspaces. */
  public static Builder opq(int subspaces) {
    return new Builder().quantizerType(QuantizerType.OPTIMIZED_PRODUCT).subspaces(subspaces);
  }

  /** Returns a builder pre-configured for sign-bit binary quantization. */
  public static Builder binarySign() {
    return new Builder().quantizerType(QuantizerType.BINARY_SIGN);
  }

  /** Returns a builder pre-configured for BBQ binary quantization. */
  public static Builder binaryBbq() {
    return new Builder().quantizerType(QuantizerType.BINARY_BBQ);
  }

  /** Returns a builder pre-configured for 1-bit RaBitQ. */
  public static Builder rabit() {
    return new Builder().quantizerType(QuantizerType.RABIT);
  }

  /** Returns a builder pre-configured for Extended RaBitQ at {@code bits} bits/dimension (2–8). */
  public static Builder extendedRabit(int bits) {
    return new Builder().quantizerType(QuantizerType.EXTENDED_RABIT).bits(bits);
  }

  /** Returns a builder pre-configured for TurboQuantizer at {@code bits} bits/dimension (1–8). */
  public static Builder turbo(int bits) {
    return new Builder().quantizerType(QuantizerType.TURBO).bits(bits);
  }

  /** Returns a builder pre-configured for NVQ with default subvector sizing. */
  public static Builder nvq() {
    return new Builder().quantizerType(QuantizerType.NVQ);
  }

  /** Returns an unconfigured builder (defaults: SCALAR_INT8, EUCLIDEAN). */
  public static Builder builder() {
    return new Builder();
  }

  // ---------------------------------------------------------------------------
  // Pipeline operations
  // ---------------------------------------------------------------------------

  /**
   * Encodes a single vector to its compressed byte representation.
   *
   * @param vector the input vector (must match the training dimension)
   * @return the compressed byte code
   */
  public byte[] encode(float[] vector) {
    return quantizer.encode(vector);
  }

  /**
   * Decodes a compressed byte code back to an approximate float vector.
   *
   * @param code the compressed byte code
   * @return the reconstructed (approximate) vector
   */
  public float[] decode(byte[] code) {
    return quantizer.decode(code);
  }

  /**
   * Compresses an entire dataset, returning a {@link CompressedVectors} suitable for approximate
   * scoring via {@link #scorerFor}.
   *
   * @param dataset the dataset to compress
   * @return compressed representation of all vectors
   */
  @SuppressWarnings("unchecked")
  public CompressedVectors compress(VectorDataset dataset) {
    return ((Quantizer<CompressedVectors>) quantizer).encodeAll(dataset);
  }

  /**
   * Returns a {@link ScoreFunction} that scores {@code query} against each ordinal in {@code
   * compressed} using the pipeline's similarity function.
   *
   * @param query the (uncompressed) query vector
   * @param compressed the corpus produced by {@link #compress}
   * @return a scorer; {@link ScoreFunction#score(int)} returns the approximate similarity
   */
  public ScoreFunction scorerFor(float[] query, CompressedVectors compressed) {
    return compressed.scoreFunctionFor(query, similarity);
  }

  /** Returns the compression ratio (uncompressed bytes ÷ compressed bytes). */
  public float compressionRatio() {
    return quantizer.compressionRatio();
  }

  /** Returns the vector dimension this pipeline was trained on. */
  public int dimension() {
    return quantizer.dimension();
  }

  /** Returns the quantizer family used by this pipeline. */
  public QuantizerType type() {
    return type;
  }

  /** Returns the underlying trained quantizer (for advanced / introspection use). */
  public Quantizer<?> quantizer() {
    return quantizer;
  }

  /** Returns the similarity function this pipeline scores against. */
  public SimilarityFunction similarity() {
    return similarity;
  }

  // ---------------------------------------------------------------------------
  // Builder
  // ---------------------------------------------------------------------------

  /**
   * Fluent builder for {@link QuantizationPipeline}.
   *
   * <p>Call {@link #train(VectorDataset)} to fit the chosen quantizer and obtain a ready pipeline.
   */
  public static final class Builder {

    private QuantizerType type = QuantizerType.SCALAR_INT8;
    private SimilarityFunction similarity = SimilarityFunction.EUCLIDEAN;
    private ScalarBits scalarBits = ScalarBits.INT8;
    private float confidence = 0.99f;
    private int subspaces = 8;
    private int clusters = 256;
    private int bits = 4;
    private int iterations = 5;
    private long seed = 42L;
    private Rotation rotation = null;

    private Builder() {}

    /** Sets the quantizer family. */
    public Builder quantizerType(QuantizerType t) {
      this.type = Objects.requireNonNull(t, "type");
      return this;
    }

    /** Sets the similarity function used for scoring (default: EUCLIDEAN). */
    public Builder similarity(SimilarityFunction sim) {
      this.similarity = Objects.requireNonNull(sim, "similarity");
      return this;
    }

    /**
     * Sets the number of PQ subspaces (used by PRODUCT and OPTIMIZED_PRODUCT). Must divide the
     * vector dimension evenly (default: 8).
     */
    public Builder subspaces(int m) {
      if (m <= 0) throw new IllegalArgumentException("subspaces must be positive: " + m);
      this.subspaces = m;
      return this;
    }

    /**
     * Sets the number of PQ codebook clusters per subspace (used by PRODUCT and OPTIMIZED_PRODUCT;
     * default: 256).
     */
    public Builder clusters(int k) {
      if (k <= 0) throw new IllegalArgumentException("clusters must be positive: " + k);
      this.clusters = k;
      return this;
    }

    /**
     * Sets the confidence interval for scalar quantization clipping (used by SCALAR_INT8 and
     * SCALAR_INT4; default: 0.99).
     */
    public Builder confidence(float c) {
      if (c <= 0f || c > 1f)
        throw new IllegalArgumentException("confidence must be in (0,1]: " + c);
      this.confidence = c;
      return this;
    }

    /**
     * Sets the bit width for EXTENDED_RABIT (2–8) and TURBO (1–8). Ignored for other types
     * (default: 4).
     */
    public Builder bits(int b) {
      if (b < 1 || b > 8) throw new IllegalArgumentException("bits must be in [1,8]: " + b);
      this.bits = b;
      return this;
    }

    /**
     * Sets the number of OPQ alternating-optimization iterations (used by OPTIMIZED_PRODUCT;
     * default: 5).
     */
    public Builder iterations(int n) {
      if (n <= 0) throw new IllegalArgumentException("iterations must be positive: " + n);
      this.iterations = n;
      return this;
    }

    /** Sets the random seed for rotation-based quantizers (default: 42). */
    public Builder seed(long s) {
      this.seed = s;
      return this;
    }

    /**
     * Sets a custom {@link Rotation} strategy for RABIT, EXTENDED_RABIT, and TURBO. When {@code
     * null} (the default), each quantizer chooses its own default rotation.
     */
    public Builder rotation(Rotation r) {
      this.rotation = r;
      return this;
    }

    /**
     * Trains the selected quantizer on {@code dataset} and returns the ready pipeline.
     *
     * @param dataset training corpus (used for codebook fitting / quantile estimation)
     * @return a trained {@link QuantizationPipeline} ready for encode/compress/score
     * @throws NullPointerException if {@code dataset} is null
     */
    public QuantizationPipeline train(VectorDataset dataset) {
      Objects.requireNonNull(dataset, "dataset");
      Quantizer<?> q =
          switch (type) {
            case SCALAR_INT8 -> ScalarQuantizer.train(dataset, ScalarBits.INT8, confidence);
            case SCALAR_INT4 -> ScalarQuantizer.train(dataset, ScalarBits.INT4, confidence);
            case PRODUCT -> ProductQuantizer.train(dataset, subspaces, clusters);
            case OPTIMIZED_PRODUCT ->
                OptimizedProductQuantizer.train(dataset, subspaces, clusters, iterations, seed);
            case BINARY_SIGN -> BinaryQuantizer.train(dataset, BinaryMode.SIGN_BIT);
            case BINARY_BBQ -> BinaryQuantizer.train(dataset, BinaryMode.BBQ);
            case RABIT ->
                rotation != null
                    ? RaBitQuantizer.train(dataset, rotation)
                    : RaBitQuantizer.train(dataset, seed);
            case EXTENDED_RABIT ->
                rotation != null
                    ? ExtendedRaBitQuantizer.train(dataset, bits, rotation)
                    : ExtendedRaBitQuantizer.train(dataset, bits, seed);
            case TURBO ->
                rotation != null
                    ? TurboQuantizer.train(dataset, bits, rotation)
                    : TurboQuantizer.train(dataset, bits, seed);
            case NVQ -> NVQuantizer.train(dataset, Math.max(1, dataset.dimension() / 64));
          };
      return new QuantizationPipeline(q, type, similarity);
    }
  }
}
