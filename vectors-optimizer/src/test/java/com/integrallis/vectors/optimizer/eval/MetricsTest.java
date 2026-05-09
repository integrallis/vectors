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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.optimizer.data.Qrels;
import com.integrallis.vectors.optimizer.data.Run;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MetricsTest {

  // Single query "q1" with three relevant docs (graded 3, 2, 1).
  // Run ranks d1 (rel=3), d2 (rel=2), d_unrelated, d3 (rel=1) in that order.
  private static Qrels singleQueryQrels() {
    return new Qrels(Map.of("q1", Map.of("d1", 3, "d2", 2, "d3", 1)));
  }

  private static Run rankedRun() {
    LinkedHashMap<String, Double> ranking = new LinkedHashMap<>();
    ranking.put("d1", 0.95);
    ranking.put("d2", 0.85);
    ranking.put("d_unrelated", 0.50);
    ranking.put("d3", 0.30);
    return new Run(Map.of("q1", ranking));
  }

  @Test
  void ndcgAtKMatchesKnownValue() {
    // DCG@4 = (2^3-1)/log2(2) + (2^2-1)/log2(3) + 0 + (2^1-1)/log2(5)
    //       = 7/1 + 3/1.5849625 + 1/2.321928
    //       = 7 + 1.892789 + 0.430676 = 9.323465
    // IDCG@4 = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    //        = 7 + 1.892789 + 1/2 = 9.392789
    // NDCG@4 ≈ 0.99262
    double ndcg = Metrics.ndcgAtK(singleQueryQrels(), rankedRun(), 4);
    assertThat(ndcg).isCloseTo(0.99262, org.assertj.core.data.Offset.offset(1e-4));
  }

  @Test
  void recallAtKHandlesEmptyRun() {
    Run empty = new Run(Map.of("q1", new LinkedHashMap<>()));
    assertThat(Metrics.recallAtK(singleQueryQrels(), empty, 10)).isZero();
  }

  @Test
  void precisionAtKDividesByK() {
    // Top-2 of rankedRun are both relevant => precision@2 = 2/2 = 1.0
    assertThat(Metrics.precisionAtK(singleQueryQrels(), rankedRun(), 2)).isEqualTo(1.0);
    // Top-3 has 1 unrelated => precision@3 = 2/3
    assertThat(Metrics.precisionAtK(singleQueryQrels(), rankedRun(), 3))
        .isCloseTo(2.0 / 3.0, org.assertj.core.data.Offset.offset(1e-9));
  }

  @Test
  void f1IsHarmonicMean() {
    // recall@2 = 2/3, precision@2 = 1.0
    // F1 = 2 * (2/3 * 1) / (2/3 + 1) = (4/3) / (5/3) = 4/5 = 0.8
    assertThat(Metrics.f1AtK(singleQueryQrels(), rankedRun(), 2))
        .isCloseTo(0.8, org.assertj.core.data.Offset.offset(1e-9));
  }

  @Test
  void mrrTakesFirstRelevantRank() {
    // d1 is relevant at rank 1 → MRR = 1.0
    assertThat(Metrics.mrr(singleQueryQrels(), rankedRun())).isEqualTo(1.0);

    // Move first relevant to rank 3
    LinkedHashMap<String, Double> rerank = new LinkedHashMap<>();
    rerank.put("d_a", 0.99);
    rerank.put("d_b", 0.98);
    rerank.put("d1", 0.97);
    Run shifted = new Run(Map.of("q1", rerank));
    assertThat(Metrics.mrr(singleQueryQrels(), shifted))
        .isCloseTo(1.0 / 3.0, org.assertj.core.data.Offset.offset(1e-9));
  }

  @Test
  void queriesWithoutPositiveJudgementsAreSkipped() {
    Qrels qrels =
        new Qrels(
            Map.of(
                "q1", Map.of("d1", 1),
                "q_blank", Map.of("d_x", 0)));
    LinkedHashMap<String, Double> r = new LinkedHashMap<>();
    r.put("d1", 1.0);
    Run run = new Run(Map.of("q1", r));
    // Recall is averaged only over q1 → 1.0, not 0.5.
    assertThat(Metrics.recallAtK(qrels, run, 5)).isEqualTo(1.0);
  }
}
