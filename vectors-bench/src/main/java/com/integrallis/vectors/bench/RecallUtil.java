package com.integrallis.vectors.bench;

import com.integrallis.vectors.core.SimilarityFunction;
import java.util.Arrays;

/**
 * Utilities for computing recall@k and brute-force ground truth used across benchmark and slow test
 * classes.
 *
 * <p>All methods are pure functions with no shared mutable state and are safe to call from multiple
 * threads concurrently.
 */
public final class RecallUtil {

  private RecallUtil() {}

  /**
   * Computes recall@k for a single query: the fraction of the true top-k neighbors that appear
   * anywhere in the approximate result list.
   *
   * @param trueNeighbors ground-truth neighbor ordinals for this query (sorted by distance, only
   *     first {@code k} entries used)
   * @param approxNeighbors approximate result ordinals returned by the index (only first {@code k}
   *     entries used)
   * @param k number of neighbors to evaluate
   * @return recall in [0.0, 1.0]
   */
  public static double recallAtK(int[] trueNeighbors, int[] approxNeighbors, int k) {
    int limit = Math.min(k, Math.min(trueNeighbors.length, approxNeighbors.length));
    int hits = 0;
    for (int i = 0; i < limit; i++) {
      int candidate = approxNeighbors[i];
      for (int j = 0; j < limit; j++) {
        if (trueNeighbors[j] == candidate) {
          hits++;
          break;
        }
      }
    }
    return (double) hits / limit;
  }

  /**
   * Computes mean recall@k across a set of queries.
   *
   * @param trueNeighbors ground-truth neighbor ordinals, one row per query
   * @param approxNeighbors approximate results, one row per query; must have the same number of
   *     rows as {@code trueNeighbors}
   * @param k number of neighbors to evaluate per query
   * @return mean recall over all queries
   * @throws IllegalArgumentException if the row counts differ
   */
  public static double meanRecallAtK(int[][] trueNeighbors, int[][] approxNeighbors, int k) {
    if (trueNeighbors.length != approxNeighbors.length) {
      throw new IllegalArgumentException(
          "Row count mismatch: trueNeighbors="
              + trueNeighbors.length
              + " approxNeighbors="
              + approxNeighbors.length);
    }
    double sum = 0.0;
    for (int i = 0; i < trueNeighbors.length; i++) {
      sum += recallAtK(trueNeighbors[i], approxNeighbors[i], k);
    }
    return sum / trueNeighbors.length;
  }

  /**
   * Computes brute-force exact top-k neighbors for a single query vector. Intended for ground-truth
   * generation in offline {@code @Setup} methods, not inside timed benchmark loops.
   *
   * <p>Ties (vectors with identical scores) are broken by ordinal to produce a deterministic
   * ordering.
   *
   * @param query the query vector
   * @param vectors the corpus; {@code vectors[i]} is the vector at ordinal {@code i}
   * @param sim the similarity function (higher score = nearer neighbor)
   * @param k number of nearest neighbors to return
   * @return array of length {@code min(k, vectors.length)} containing ordinals of the top-k nearest
   *     neighbors, ordered by descending score
   */
  public static int[] bruteForceKnn(
      float[] query, float[][] vectors, SimilarityFunction sim, int k) {
    int n = vectors.length;
    // Score every vector.
    float[] scores = new float[n];
    for (int i = 0; i < n; i++) {
      scores[i] = sim.compare(query, vectors[i]);
    }
    // Partial sort: find the top-k by ordinal using an index sort.
    Integer[] ordinals = new Integer[n];
    for (int i = 0; i < n; i++) ordinals[i] = i;
    // Sort descending by score; break ties ascending by ordinal (deterministic).
    Arrays.sort(
        ordinals,
        (a, b) -> {
          int cmp = Float.compare(scores[b], scores[a]);
          return cmp != 0 ? cmp : Integer.compare(a, b);
        });
    int limit = Math.min(k, n);
    int[] result = new int[limit];
    for (int i = 0; i < limit; i++) result[i] = ordinals[i];
    return result;
  }

  /**
   * Computes brute-force ground truth for all queries against a corpus.
   *
   * @param queries query vectors
   * @param corpus base vectors
   * @param sim the similarity function
   * @param k number of nearest neighbors per query
   * @return ground-truth array, one row of length {@code k} per query
   */
  public static int[][] bruteForceGroundTruth(
      float[][] queries, float[][] corpus, SimilarityFunction sim, int k) {
    int[][] gt = new int[queries.length][];
    for (int i = 0; i < queries.length; i++) {
      gt[i] = bruteForceKnn(queries[i], corpus, sim, k);
    }
    return gt;
  }
}
