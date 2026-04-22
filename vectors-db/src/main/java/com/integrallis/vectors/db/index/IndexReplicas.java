package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntPredicate;

/**
 * Composition wrapper that replicates the corpus across {@code N} identical {@link IndexSpi}
 * replicas and round-robins queries across them. Intended as a simple horizontal-concurrency
 * primitive for backends whose single-query path holds a monitor or shared scratch pool: with
 * {@code N} replicas the effective concurrent-search capacity scales up to {@code N} independent
 * search streams without coordination between them. Modelled on FAISS {@code IndexReplicas}.
 *
 * <p><b>Build.</b> {@link #build} is invoked on every replica with the same input, sequentially.
 * All replicas therefore hold identical ordinals — a search against any replica returns ordinals
 * that are directly comparable with any other replica and with the logical corpus.
 *
 * <p><b>Selection policy.</b> Round-robin via an {@link AtomicInteger} cursor; lock-free. Every
 * call to {@link #search} advances the cursor once, so back-to-back calls on a shared {@code
 * IndexReplicas} instance distribute across replicas even from a single caller thread.
 *
 * <p><b>Thread safety.</b> Concurrent {@link #search} is safe if each underlying replica is safe
 * for concurrent search; replicas receive disjoint subsets of the query stream thanks to
 * round-robin dispatch. Concurrent {@link #build} is not safe.
 */
public final class IndexReplicas implements IndexSpi {

  private final List<IndexSpi> replicas;
  private final AtomicInteger cursor = new AtomicInteger();
  private int commonSize;

  /**
   * @param replicas non-empty list of SPIs; each must accept the same vectors during {@link #build}
   */
  public IndexReplicas(List<IndexSpi> replicas) {
    Objects.requireNonNull(replicas, "replicas must not be null");
    if (replicas.isEmpty()) {
      throw new IllegalArgumentException("replicas must not be empty");
    }
    for (int i = 0; i < replicas.size(); i++) {
      if (replicas.get(i) == null) {
        throw new IllegalArgumentException("replica " + i + " must not be null");
      }
    }
    this.replicas = List.copyOf(replicas);
  }

  @Override
  public void build(float[][] vectors, SimilarityFunction metric) {
    Objects.requireNonNull(vectors, "vectors must not be null");
    Objects.requireNonNull(metric, "metric must not be null");
    for (IndexSpi r : replicas) {
      r.build(vectors, metric);
    }
    this.commonSize = vectors.length;
  }

  @Override
  public SearchOutcome search(float[] query, int k, int searchListSize, float overQueryFactor) {
    return next().search(query, k, searchListSize, overQueryFactor);
  }

  @Override
  public SearchOutcome search(
      float[] query, int k, int searchListSize, float overQueryFactor, int searchMultiStart) {
    return next().search(query, k, searchListSize, overQueryFactor, searchMultiStart);
  }

  @Override
  public SearchOutcome searchWithPredicate(
      float[] query, int k, int searchListSize, float overQueryFactor, IntPredicate predicate) {
    return next().searchWithPredicate(query, k, searchListSize, overQueryFactor, predicate);
  }

  @Override
  public int size() {
    return commonSize;
  }

  @Override
  public void close() {
    for (IndexSpi r : replicas) {
      r.close();
    }
  }

  /** Returns the next replica using atomic round-robin dispatch. */
  private IndexSpi next() {
    int idx = Math.floorMod(cursor.getAndIncrement(), replicas.size());
    return replicas.get(idx);
  }
}
