package com.integrallis.vectors.quantization;

import com.integrallis.vectors.core.SimilarityFunction;

/**
 * Compressed vector storage with approximate scoring. Created by {@link Quantizer#encodeAll}.
 *
 * <p>Holds the compressed representation of a set of vectors and can create {@link ScoreFunction}
 * instances for fast approximate similarity queries.
 */
public interface CompressedVectors extends AutoCloseable {

  /** Returns the number of stored vectors. */
  int size();

  /** Returns the original dimensionality of the vectors before compression. */
  int dimension();

  /**
   * Creates a score function for the given query vector. The returned function computes approximate
   * similarity scores against stored vectors by ordinal.
   *
   * <p>The returned {@link ScoreFunction} is not thread-safe; each thread should create its own
   * instance.
   *
   * @param query the query vector (full-precision float)
   * @param similarityFunction the similarity metric to use
   * @return a function that scores stored vectors against the query
   */
  ScoreFunction scoreFunctionFor(float[] query, SimilarityFunction similarityFunction);

  @Override
  void close();
}
