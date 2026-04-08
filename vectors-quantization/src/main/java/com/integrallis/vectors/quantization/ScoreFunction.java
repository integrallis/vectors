package com.integrallis.vectors.quantization;

/**
 * Computes the approximate similarity score between a query and a stored vector identified by
 * ordinal. Created by {@link CompressedVectors#scoreFunctionFor}.
 *
 * <p>Implementations are not thread-safe; each thread should create its own instance via {@link
 * CompressedVectors#scoreFunctionFor}.
 */
@FunctionalInterface
public interface ScoreFunction {

  /**
   * Returns the approximate similarity score for the vector at the given ordinal.
   *
   * @param ordinal the 0-based index of the stored vector
   * @return the approximate similarity score (higher means more similar)
   */
  float score(int ordinal);
}
