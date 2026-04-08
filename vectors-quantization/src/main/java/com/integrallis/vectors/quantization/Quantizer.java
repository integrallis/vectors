package com.integrallis.vectors.quantization;

/**
 * Encodes vectors into compressed representations that support fast approximate scoring. A
 * quantizer learns compression parameters (quantiles, codebooks, etc.) during training and then
 * encodes vectors into a compact form.
 *
 * <p>Training is implementation-specific (e.g., {@code ScalarQuantizer.train(dataset)}). After
 * training, the quantizer holds its learned state and can encode vectors.
 *
 * @param <C> the type of compressed vectors produced by this quantizer
 */
public interface Quantizer<C extends CompressedVectors> {

  /**
   * Encodes a single vector into its compressed byte representation.
   *
   * @param vector the input float vector
   * @return the compressed byte array
   */
  byte[] encode(float[] vector);

  /**
   * Encodes a single vector into the destination byte array, returning the per-vector correction
   * factor (if applicable, 0.0 otherwise).
   *
   * @param vector the input float vector
   * @param dst the destination byte array (must be large enough)
   * @return the correction factor for this vector
   */
  float encode(float[] vector, byte[] dst);

  /**
   * Decodes a compressed byte representation back to an approximate float vector.
   *
   * @param encoded the compressed byte array
   * @return the reconstructed float vector
   */
  float[] decode(byte[] encoded);

  /**
   * Encodes all vectors in the dataset into compressed storage.
   *
   * @param dataset the source vectors
   * @return compressed vectors with approximate scoring capability
   */
  C encodeAll(VectorDataset dataset);

  /**
   * Returns the compression ratio (original bytes / compressed bytes per vector). For example, 4.0
   * means 4x compression (float32 → int8).
   */
  float compressionRatio();

  /** Returns the expected dimensionality of input vectors. */
  int dimension();
}
