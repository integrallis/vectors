package com.integrallis.vectors.db.index;

import com.integrallis.vectors.core.SimilarityFunction;
import com.nvidia.cuvs.SearchResults;
import java.util.Map;

/**
 * Shared decoder that converts a cuVS {@link SearchResults} instance (map of neighbour-id to raw
 * distance) into an {@link IndexSpi.SearchOutcome} (parallel {@code int[]} ordinals and {@code
 * float[]} scores, descending).
 *
 * <p>cuVS reports raw distances where <em>lower is closer</em>. The rest of java-vectors orders
 * search results by descending similarity, so L2 and cosine are negated and inner-product scores
 * are returned as-is (cuVS already returns them as similarities).
 */
final class CuVsResultDecoder {

  private CuVsResultDecoder() {}

  static IndexSpi.SearchOutcome decode(SearchResults results, SimilarityFunction metric) {
    var rows = results.getResults();
    if (rows.isEmpty()) {
      return new IndexSpi.SearchOutcome(new int[0], new float[0]);
    }
    Map<Integer, Float> row = rows.get(0);
    int n = row.size();
    int[] ids = new int[n];
    float[] scores = new float[n];
    int i = 0;
    for (Map.Entry<Integer, Float> e : row.entrySet()) {
      ids[i] = e.getKey();
      scores[i] = toScore(metric, e.getValue());
      i++;
    }
    sortByScoreDescending(ids, scores);
    return new IndexSpi.SearchOutcome(ids, scores);
  }

  private static float toScore(SimilarityFunction metric, float rawDistance) {
    return switch (metric) {
      case EUCLIDEAN, COSINE -> -rawDistance;
      case DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> rawDistance;
    };
  }

  private static void sortByScoreDescending(int[] ids, float[] scores) {
    // Insertion sort: result sets returned by cuVS are small (top-k, typically <=100).
    for (int i = 1; i < scores.length; i++) {
      float score = scores[i];
      int id = ids[i];
      int j = i - 1;
      while (j >= 0 && scores[j] < score) {
        scores[j + 1] = scores[j];
        ids[j + 1] = ids[j];
        j--;
      }
      scores[j + 1] = score;
      ids[j + 1] = id;
    }
  }
}
