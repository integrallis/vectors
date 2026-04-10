package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;

/**
 * Internal index SPI. Implementations wrap a concrete backend (flat scan, HNSW, Vamana, IVF) and
 * expose a uniform search API to the {@link com.integrallis.vectors.db.VectorCollection} facade.
 *
 * <p><b>Filter contract:</b> {@code search()} does <b>not</b> take a filter. The post-filter is
 * applied by the {@code VectorCollection} layer <i>after</i> the SPI returns candidates. This
 * matches the design doc §6.5 decision that filter execution is not part of the SPI in v0.1.
 */
public interface IndexSpi extends AutoCloseable {

  /**
   * Builds the index from scratch from the provided vectors. The vectors are indexed in ordinal
   * order: ordinal {@code i} corresponds to {@code vectors[i]}.
   *
   * @param vectors the vectors to index (dimension must match all rows)
   * @param metric similarity function for the index
   */
  void build(float[][] vectors, SimilarityFunction metric);

  /**
   * Searches the index for the top-{@code k} nearest neighbours.
   *
   * <p><b>Parameter contract.</b> {@code searchListSize} and {@code overQueryFactor} are hints for
   * graph-based backends (HNSW, Vamana) that do a coarse pass followed by a rescore. Brute- force
   * implementations such as {@link FlatScanAdapter} already compare against every stored vector and
   * <i>ignore</i> both parameters — they return identical results regardless of what is passed.
   * Callers that mix backends should not rely on these parameters to vary flat-scan output.
   *
   * @param query the query vector
   * @param k number of final results requested
   * @param searchListSize coarse-pass beam width (may be ignored by brute-force backends)
   * @param overQueryFactor multiplier applied to {@code k} for the coarse pass; expressed as a
   *     {@code float} so non-integer factors like {@code 1.5f} or {@code 2.5f} are supported (may
   *     be ignored by brute-force backends)
   * @return the top-k ordinals and scores (descending)
   */
  SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor);

  /** Returns the number of vectors currently in the index. */
  int size();

  /** Releases any resources held by the SPI. Default: no-op. */
  @Override
  default void close() {}

  /** Internal carrier for the raw search output (ordinals + scores, descending). */
  record SearchOutcome(int[] ordinals, float[] scores) {
    public SearchOutcome {
      if (ordinals.length != scores.length) {
        throw new IllegalArgumentException(
            "ordinals.length ("
                + ordinals.length
                + ") must match scores.length ("
                + scores.length
                + ")");
      }
    }
  }
}
