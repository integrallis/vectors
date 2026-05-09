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
package com.integrallis.vectors.optimizer.objective;

/**
 * Combines per-axis metrics into a single scalar to be maximized.
 *
 * <p>Quality axes (recall, NDCG, precision, F1, MRR) are already in [0, 1] and contribute
 * positively. Cost axes (p95 latency, build time, physical memory) are normalized by the
 * corresponding {@code *Reference} fields of {@link ObjectiveWeights}, clipped to [0, 1] and
 * subtracted. The result is the weighted sum of these signed contributions.
 */
public final class Objective {

  private Objective() {}

  /**
   * Computes the composite score from raw metric values.
   *
   * @param recall Recall@k in [0, 1]
   * @param ndcg NDCG@k in [0, 1]
   * @param precision Precision@k in [0, 1]
   * @param f1 F1@k in [0, 1]
   * @param mrr MRR in [0, 1]
   * @param latencyP95Us p95 search latency in microseconds (MIN)
   * @param buildTimeMs collection build time in milliseconds (MIN)
   * @param memoryBytes physical memory footprint in bytes (MIN)
   * @param w weights and reference scales
   * @return composite scalar (higher is better)
   */
  public static double score(
      double recall,
      double ndcg,
      double precision,
      double f1,
      double mrr,
      double latencyP95Us,
      long buildTimeMs,
      long memoryBytes,
      ObjectiveWeights w) {
    double s = 0.0;
    s += w.recallWeight() * recall;
    s += w.ndcgWeight() * ndcg;
    s += w.precisionWeight() * precision;
    s += w.f1Weight() * f1;
    s += w.mrrWeight() * mrr;
    if (w.latencyP95Weight() > 0) {
      s -= w.latencyP95Weight() * clip01(latencyP95Us / w.latencyP95ReferenceUs());
    }
    if (w.buildTimeWeight() > 0) {
      s -= w.buildTimeWeight() * clip01((double) buildTimeMs / w.buildTimeReferenceMs());
    }
    if (w.memoryWeight() > 0) {
      s -= w.memoryWeight() * clip01((double) memoryBytes / w.memoryReferenceBytes());
    }
    return s;
  }

  private static double clip01(double v) {
    if (Double.isNaN(v) || v < 0.0) return 0.0;
    return Math.min(1.0, v);
  }
}
