/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.vectors.optimizer.eval;

import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Run;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure-function IR metrics over {@link Qrels} + {@link Run}.
 *
 * <p>NDCG follows Järvelin &amp; Kekäläinen (2002): {@code DCG = sum_i (2^rel_i - 1) / log2(i+1)}
 * with {@code i} starting at 1; the IDCG is computed against the same query's optimal ranking of
 * its judged docs. Recall@k, Precision@k, F1@k and MRR follow standard textbook definitions.
 *
 * <p>Each metric is averaged macro across queries that have at least one positively-graded
 * judgement. Queries with no positive grades are skipped (they cannot contribute meaningful
 * recall).
 */
public final class Metrics {

  private static final double LN2 = Math.log(2.0);

  private Metrics() {}

  /** NDCG@k averaged over queries. */
  public static double ndcgAtK(Qrels q, Run r, int k) {
    requirePositiveK(k);
    double sum = 0.0;
    int counted = 0;
    for (var entry : q.relevance().entrySet()) {
      if (!hasPositive(entry.getValue())) continue;
      LinkedHashMap<String, Double> ranked =
          r.ranking().getOrDefault(entry.getKey(), new LinkedHashMap<>());
      double dcg = dcg(ranked, entry.getValue(), k);
      double idcg = idealDcg(entry.getValue(), k);
      sum += idcg == 0.0 ? 0.0 : dcg / idcg;
      counted++;
    }
    return counted == 0 ? 0.0 : sum / counted;
  }

  /** Recall@k averaged over queries: |relevant ∩ top-k| / |relevant|. */
  public static double recallAtK(Qrels q, Run r, int k) {
    requirePositiveK(k);
    double sum = 0.0;
    int counted = 0;
    for (var entry : q.relevance().entrySet()) {
      int totalRel = countPositive(entry.getValue());
      if (totalRel == 0) continue;
      int hits = countHits(r.ranking().get(entry.getKey()), entry.getValue(), k);
      sum += (double) hits / totalRel;
      counted++;
    }
    return counted == 0 ? 0.0 : sum / counted;
  }

  /** Precision@k averaged over queries: |relevant ∩ top-k| / k. */
  public static double precisionAtK(Qrels q, Run r, int k) {
    requirePositiveK(k);
    double sum = 0.0;
    int counted = 0;
    for (var entry : q.relevance().entrySet()) {
      if (!hasPositive(entry.getValue())) continue;
      int hits = countHits(r.ranking().get(entry.getKey()), entry.getValue(), k);
      sum += (double) hits / k;
      counted++;
    }
    return counted == 0 ? 0.0 : sum / counted;
  }

  /** F1@k = harmonic mean of recall@k and precision@k (computed per-query then averaged). */
  public static double f1AtK(Qrels q, Run r, int k) {
    requirePositiveK(k);
    double sum = 0.0;
    int counted = 0;
    for (var entry : q.relevance().entrySet()) {
      int totalRel = countPositive(entry.getValue());
      if (totalRel == 0) continue;
      int hits = countHits(r.ranking().get(entry.getKey()), entry.getValue(), k);
      double recall = (double) hits / totalRel;
      double precision = (double) hits / k;
      double denom = recall + precision;
      sum += denom == 0.0 ? 0.0 : 2.0 * recall * precision / denom;
      counted++;
    }
    return counted == 0 ? 0.0 : sum / counted;
  }

  /** Mean Reciprocal Rank: 1 / rank of first relevant doc, averaged over queries. */
  public static double mrr(Qrels q, Run r) {
    double sum = 0.0;
    int counted = 0;
    for (var entry : q.relevance().entrySet()) {
      if (!hasPositive(entry.getValue())) continue;
      LinkedHashMap<String, Double> ranked = r.ranking().get(entry.getKey());
      double rr = 0.0;
      if (ranked != null) {
        int rank = 1;
        for (String docId : ranked.keySet()) {
          Integer grade = entry.getValue().get(docId);
          if (grade != null && grade > 0) {
            rr = 1.0 / rank;
            break;
          }
          rank++;
        }
      }
      sum += rr;
      counted++;
    }
    return counted == 0 ? 0.0 : sum / counted;
  }

  private static double dcg(LinkedHashMap<String, Double> ranked, Map<String, Integer> rel, int k) {
    if (ranked == null || ranked.isEmpty()) return 0.0;
    double dcg = 0.0;
    int rank = 1;
    for (String docId : ranked.keySet()) {
      if (rank > k) break;
      Integer grade = rel.get(docId);
      if (grade != null && grade > 0) {
        dcg += (Math.pow(2.0, grade) - 1.0) / (Math.log(rank + 1.0) / LN2);
      }
      rank++;
    }
    return dcg;
  }

  private static double idealDcg(Map<String, Integer> rel, int k) {
    List<Integer> grades = new ArrayList<>(rel.values());
    grades.sort(Comparator.reverseOrder());
    double idcg = 0.0;
    for (int i = 0; i < Math.min(k, grades.size()); i++) {
      int g = grades.get(i);
      if (g <= 0) break;
      idcg += (Math.pow(2.0, g) - 1.0) / (Math.log(i + 2.0) / LN2);
    }
    return idcg;
  }

  private static int countHits(
      LinkedHashMap<String, Double> ranked, Map<String, Integer> rel, int k) {
    if (ranked == null) return 0;
    int hits = 0;
    int rank = 0;
    for (String docId : ranked.keySet()) {
      if (rank >= k) break;
      Integer grade = rel.get(docId);
      if (grade != null && grade > 0) hits++;
      rank++;
    }
    return hits;
  }

  private static int countPositive(Map<String, Integer> rel) {
    int n = 0;
    for (int g : rel.values()) if (g > 0) n++;
    return n;
  }

  private static boolean hasPositive(Map<String, Integer> rel) {
    for (int g : rel.values()) if (g > 0) return true;
    return false;
  }

  private static void requirePositiveK(int k) {
    if (k <= 0) throw new IllegalArgumentException("k must be > 0 (was " + k + ")");
  }
}
