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
package com.integrallis.vectors.bench.report;

import java.util.Map;
import java.util.Objects;

/**
 * Immutable container for a single benchmark measurement point.
 *
 * <p>Captures the dataset, algorithm, parameter configuration, and all measured metrics for one
 * (build-config, search-config) pair. Collections of results across a parameter sweep form a
 * recall-QPS curve.
 *
 * @param dataset the dataset name (e.g., "sift-128-euclidean")
 * @param algorithm the algorithm name (e.g., "hnsw", "vamana", "ivf_flat", "flat")
 * @param buildParams build-time parameters (e.g., M=32, efConstruction=200)
 * @param searchParams search-time parameters (e.g., efSearch=128)
 * @param recall10 recall@10 in [0, 1]
 * @param recall100 recall@100 in [0, 1], or -1 if not measured
 * @param qps queries per second (sequential, single-threaded)
 * @param p50Us median query latency in microseconds
 * @param p95Us 95th percentile query latency in microseconds
 * @param p99Us 99th percentile query latency in microseconds
 * @param buildTimeSeconds wall-clock index construction time in seconds
 * @param indexSizeMb total on-disk or in-memory index size in megabytes
 * @param compressionRatio original size / compressed size, or 1.0 for uncompressed
 * @param extra additional metrics (encode_us, score_us, memory_mb, etc.)
 */
public record BenchmarkResult(
    String dataset,
    String algorithm,
    Map<String, String> buildParams,
    Map<String, String> searchParams,
    double recall10,
    double recall100,
    double qps,
    double p50Us,
    double p95Us,
    double p99Us,
    double buildTimeSeconds,
    double indexSizeMb,
    double compressionRatio,
    Map<String, String> extra) {

  public BenchmarkResult {
    Objects.requireNonNull(dataset, "dataset");
    Objects.requireNonNull(algorithm, "algorithm");
    buildParams = Map.copyOf(buildParams);
    searchParams = Map.copyOf(searchParams);
    extra = Map.copyOf(extra);
  }

  /** Creates a builder for constructing results. */
  public static Builder builder(String dataset, String algorithm) {
    return new Builder(dataset, algorithm);
  }

  /** Builder for {@link BenchmarkResult}. */
  public static final class Builder {
    private final String dataset;
    private final String algorithm;
    private Map<String, String> buildParams = Map.of();
    private Map<String, String> searchParams = Map.of();
    private double recall10;
    private double recall100 = -1;
    private double qps;
    private double p50Us;
    private double p95Us;
    private double p99Us;
    private double buildTimeSeconds;
    private double indexSizeMb;
    private double compressionRatio = 1.0;
    private Map<String, String> extra = Map.of();

    private Builder(String dataset, String algorithm) {
      this.dataset = Objects.requireNonNull(dataset);
      this.algorithm = Objects.requireNonNull(algorithm);
    }

    public Builder buildParams(Map<String, String> v) {
      this.buildParams = v;
      return this;
    }

    public Builder searchParams(Map<String, String> v) {
      this.searchParams = v;
      return this;
    }

    public Builder recall10(double v) {
      this.recall10 = v;
      return this;
    }

    public Builder recall100(double v) {
      this.recall100 = v;
      return this;
    }

    public Builder qps(double v) {
      this.qps = v;
      return this;
    }

    public Builder p50Us(double v) {
      this.p50Us = v;
      return this;
    }

    public Builder p95Us(double v) {
      this.p95Us = v;
      return this;
    }

    public Builder p99Us(double v) {
      this.p99Us = v;
      return this;
    }

    public Builder buildTimeSeconds(double v) {
      this.buildTimeSeconds = v;
      return this;
    }

    public Builder indexSizeMb(double v) {
      this.indexSizeMb = v;
      return this;
    }

    public Builder compressionRatio(double v) {
      this.compressionRatio = v;
      return this;
    }

    public Builder extra(Map<String, String> v) {
      this.extra = v;
      return this;
    }

    public BenchmarkResult build() {
      return new BenchmarkResult(
          dataset,
          algorithm,
          buildParams,
          searchParams,
          recall10,
          recall100,
          qps,
          p50Us,
          p95Us,
          p99Us,
          buildTimeSeconds,
          indexSizeMb,
          compressionRatio,
          extra);
    }
  }
}
