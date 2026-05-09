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
 * Per-axis weights and reference scales for {@link Objective}.
 *
 * <p>The retrieval-quality axes ({@code recall}, {@code ndcg}, {@code precision}, {@code f1},
 * {@code mrr}) are already in [0, 1] and treated as MAXIMIZE. The cost axes — latency, build time
 * and memory — are normalized by the {@code *Reference} fields and treated as MINIMIZE: each
 * normalized cost is subtracted from the composite. Any axis whose weight is {@code 0} drops out
 * of the composite entirely.
 *
 * @param kForMetrics k passed to the {@code @K} metrics
 * @param recallWeight weight for Recall@k (MAX)
 * @param ndcgWeight weight for NDCG@k (MAX)
 * @param precisionWeight weight for Precision@k (MAX)
 * @param f1Weight weight for F1@k (MAX)
 * @param mrrWeight weight for MRR (MAX)
 * @param latencyP95Weight weight for p95 search latency (MIN)
 * @param latencyP95ReferenceUs reference latency (µs) used to normalize p95 to [0, 1]
 * @param buildTimeWeight weight for collection build time (MIN)
 * @param buildTimeReferenceMs reference build time (ms) used to normalize to [0, 1]
 * @param memoryWeight weight for physical memory footprint (MIN)
 * @param memoryReferenceBytes reference memory (bytes) used to normalize to [0, 1]
 */
public record ObjectiveWeights(
    int kForMetrics,
    double recallWeight,
    double ndcgWeight,
    double precisionWeight,
    double f1Weight,
    double mrrWeight,
    double latencyP95Weight,
    double latencyP95ReferenceUs,
    double buildTimeWeight,
    double buildTimeReferenceMs,
    double memoryWeight,
    double memoryReferenceBytes) {

  public ObjectiveWeights {
    if (kForMetrics <= 0) {
      throw new IllegalArgumentException("kForMetrics must be > 0 (was " + kForMetrics + ")");
    }
    requireNonNegative("recallWeight", recallWeight);
    requireNonNegative("ndcgWeight", ndcgWeight);
    requireNonNegative("precisionWeight", precisionWeight);
    requireNonNegative("f1Weight", f1Weight);
    requireNonNegative("mrrWeight", mrrWeight);
    requireNonNegative("latencyP95Weight", latencyP95Weight);
    requireNonNegative("buildTimeWeight", buildTimeWeight);
    requireNonNegative("memoryWeight", memoryWeight);
    if (latencyP95Weight > 0 && latencyP95ReferenceUs <= 0) {
      throw new IllegalArgumentException("latencyP95ReferenceUs must be > 0 when weighted");
    }
    if (buildTimeWeight > 0 && buildTimeReferenceMs <= 0) {
      throw new IllegalArgumentException("buildTimeReferenceMs must be > 0 when weighted");
    }
    if (memoryWeight > 0 && memoryReferenceBytes <= 0) {
      throw new IllegalArgumentException("memoryReferenceBytes must be > 0 when weighted");
    }
  }

  private static void requireNonNegative(String name, double v) {
    if (v < 0 || Double.isNaN(v)) {
      throw new IllegalArgumentException(name + " must be >= 0 (was " + v + ")");
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Mutable builder for {@link ObjectiveWeights}. */
  public static final class Builder {
    private int kForMetrics = 10;
    private double recallWeight = 1.0;
    private double ndcgWeight;
    private double precisionWeight;
    private double f1Weight;
    private double mrrWeight;
    private double latencyP95Weight;
    private double latencyP95ReferenceUs = 1_000.0;
    private double buildTimeWeight;
    private double buildTimeReferenceMs = 60_000.0;
    private double memoryWeight;
    private double memoryReferenceBytes = 1024L * 1024L * 1024L;

    public Builder kForMetrics(int v) { this.kForMetrics = v; return this; }
    public Builder recallWeight(double v) { this.recallWeight = v; return this; }
    public Builder ndcgWeight(double v) { this.ndcgWeight = v; return this; }
    public Builder precisionWeight(double v) { this.precisionWeight = v; return this; }
    public Builder f1Weight(double v) { this.f1Weight = v; return this; }
    public Builder mrrWeight(double v) { this.mrrWeight = v; return this; }
    public Builder latencyP95Weight(double v) { this.latencyP95Weight = v; return this; }
    public Builder latencyP95ReferenceUs(double v) { this.latencyP95ReferenceUs = v; return this; }
    public Builder buildTimeWeight(double v) { this.buildTimeWeight = v; return this; }
    public Builder buildTimeReferenceMs(double v) { this.buildTimeReferenceMs = v; return this; }
    public Builder memoryWeight(double v) { this.memoryWeight = v; return this; }
    public Builder memoryReferenceBytes(double v) { this.memoryReferenceBytes = v; return this; }

    public ObjectiveWeights build() {
      return new ObjectiveWeights(
          kForMetrics,
          recallWeight,
          ndcgWeight,
          precisionWeight,
          f1Weight,
          mrrWeight,
          latencyP95Weight,
          latencyP95ReferenceUs,
          buildTimeWeight,
          buildTimeReferenceMs,
          memoryWeight,
          memoryReferenceBytes);
    }
  }
}
