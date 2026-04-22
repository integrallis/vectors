package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * Composition wrapper that turns an approximate {@link IndexSpi} into a two-pass rescore pipeline:
 * the base SPI produces a coarse candidate list of size {@code ⌈k × kFactor⌉}, and those candidates
 * are re-scored against the full-precision vectors retained by this wrapper. The final top-{@code
 * k} is returned in descending score order.
 *
 * <p>Typical use is to wrap a quantized navigator (PQ / BQ) with full-precision rescoring: the
 * approximate pass narrows the search to a small candidate window; the rescore pass restores recall
 * using exact distances. Modelled on FAISS {@code IndexRefine}.
 *
 * <p><b>Memory.</b> This wrapper retains a reference to the {@code float[][]} passed to {@link
 * #build} for the lifetime of the index. The base SPI may or may not retain its own copy; this is
 * orthogonal.
 *
 * <p><b>Thread safety.</b> Concurrent {@link #search} calls are safe as long as the wrapped base
 * SPI is safe for concurrent search (the wrapper itself is stateless during search). Concurrent
 * {@link #build} calls are not safe.
 */
public final class IndexRefine implements IndexSpi {

  private final IndexSpi base;
  private final float kFactor;

  private float[][] vectors = new float[0][];
  private SimilarityFunction metric;

  /**
   * @param base approximate backend (e.g. HNSW/Vamana over quantized vectors)
   * @param kFactor over-query multiplier applied to {@code k} for the coarse pass; must be {@code
   *     >= 1.0f}. A value of {@code 1.0f} makes the refine pass a pure rescore with no expansion;
   *     typical production values are {@code 2.0f}–{@code 4.0f}.
   */
  public IndexRefine(IndexSpi base, float kFactor) {
    this.base = Objects.requireNonNull(base, "base must not be null");
    if (kFactor < 1.0f) {
      throw new IllegalArgumentException("kFactor must be >= 1.0: " + kFactor);
    }
    this.kFactor = kFactor;
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    this.vectors = vectors;
    this.metric = metric;
    base.build(vectors, metric);
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    return rescore(base.search(query, coarseK(k), searchListSize, overQueryFactor), query, k);
  }

  @Override
  public SearchOutcome search(
      float[] query, int k, int searchListSize, float overQueryFactor, int searchMultiStart) {
    return rescore(
        base.search(query, coarseK(k), searchListSize, overQueryFactor, searchMultiStart),
        query,
        k);
  }

  @Override
  public SearchOutcome searchWithPredicate(
      float[] query, int k, int searchListSize, float overQueryFactor, IntPredicate predicate) {
    return rescore(
        base.searchWithPredicate(query, coarseK(k), searchListSize, overQueryFactor, predicate),
        query,
        k);
  }

  @Override
  public int size() {
    return vectors.length;
  }

  @Override
  public void close() {
    base.close();
  }

  private int coarseK(int k) {
    if (k <= 0) throw new IllegalArgumentException("k must be positive: " + k);
    return Math.max(k, (int) Math.ceil(k * (double) kFactor));
  }

  /** Rescore candidate ordinals against the retained full-precision vectors, returning top-k. */
  private SearchOutcome rescore(SearchOutcome coarse, float[] query, int k) {
    int n = coarse.ordinals().length;
    if (n == 0) {
      return new SearchOutcome(new int[0], new float[0]);
    }
    int[] ids = coarse.ordinals().clone();
    float[] scores = new float[n];
    for (int i = 0; i < n; i++) {
      scores[i] = metric.compare(query, vectors[ids[i]]);
    }
    // Selection sort up to min(k, n) — n is already bounded by coarseK(k), typically small.
    int finalK = Math.min(k, n);
    for (int i = 0; i < finalK; i++) {
      int bestIdx = i;
      float bestScore = scores[i];
      for (int j = i + 1; j < n; j++) {
        if (scores[j] > bestScore) {
          bestScore = scores[j];
          bestIdx = j;
        }
      }
      if (bestIdx != i) {
        float ts = scores[i];
        int ti = ids[i];
        scores[i] = scores[bestIdx];
        ids[i] = ids[bestIdx];
        scores[bestIdx] = ts;
        ids[bestIdx] = ti;
      }
    }
    if (finalK == n) {
      return new SearchOutcome(ids, scores);
    }
    int[] outIds = new int[finalK];
    float[] outScores = new float[finalK];
    System.arraycopy(ids, 0, outIds, 0, finalK);
    System.arraycopy(scores, 0, outScores, 0, finalK);
    return new SearchOutcome(outIds, outScores);
  }
}
