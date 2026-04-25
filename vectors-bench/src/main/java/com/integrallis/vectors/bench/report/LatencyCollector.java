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

import java.util.Arrays;

/**
 * Collects per-query latency samples and computes percentiles.
 *
 * <p>Stores raw nanosecond timestamps and computes p50/p95/p99/p999 from the sorted distribution.
 * This avoids the precision loss of Welford's algorithm for percentiles while maintaining O(n)
 * memory.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * LatencyCollector lc = new LatencyCollector(10_000);
 * for (float[] query : queries) {
 *   long t0 = System.nanoTime();
 *   index.search(query, k);
 *   lc.record(System.nanoTime() - t0);
 * }
 * lc.compute();
 * double p99 = lc.p99Us();
 * }</pre>
 *
 * <p>Not thread-safe — designed for single-threaded sequential query measurement per the
 * ANN-Benchmarks protocol.
 */
public final class LatencyCollector {

  private long[] samples;
  private int count;

  // Computed values (valid after compute()).
  private double p50Us;
  private double p95Us;
  private double p99Us;
  private double p999Us;
  private double meanUs;
  private double qps;

  /**
   * Creates a collector with pre-allocated capacity.
   *
   * @param capacity expected number of samples
   */
  public LatencyCollector(int capacity) {
    this.samples = new long[capacity];
  }

  /**
   * Records a single query latency in nanoseconds.
   *
   * @param nanos elapsed time for one query
   */
  public void record(long nanos) {
    if (count == samples.length) {
      samples = Arrays.copyOf(samples, samples.length * 2);
    }
    samples[count++] = nanos;
  }

  /** Resets the collector for reuse. */
  public void reset() {
    count = 0;
    p50Us = p95Us = p99Us = p999Us = meanUs = qps = 0;
  }

  /**
   * Computes percentiles and QPS from collected samples. Must be called after all samples are
   * recorded and before reading percentile values.
   */
  public void compute() {
    if (count == 0) return;

    Arrays.sort(samples, 0, count);

    p50Us = percentile(0.50);
    p95Us = percentile(0.95);
    p99Us = percentile(0.99);
    p999Us = percentile(0.999);

    long totalNanos = 0;
    for (int i = 0; i < count; i++) {
      totalNanos += samples[i];
    }
    meanUs = (totalNanos / (double) count) / 1_000.0;
    qps = count / (totalNanos / 1_000_000_000.0);
  }

  /** Returns the p50 (median) latency in microseconds. */
  public double p50Us() {
    return p50Us;
  }

  /** Returns the p95 latency in microseconds. */
  public double p95Us() {
    return p95Us;
  }

  /** Returns the p99 latency in microseconds. */
  public double p99Us() {
    return p99Us;
  }

  /** Returns the p999 latency in microseconds. */
  public double p999Us() {
    return p999Us;
  }

  /** Returns the mean latency in microseconds. */
  public double meanUs() {
    return meanUs;
  }

  /** Returns queries per second. */
  public double qps() {
    return qps;
  }

  /** Returns the number of samples recorded. */
  public int count() {
    return count;
  }

  private double percentile(double p) {
    double idx = p * (count - 1);
    int lo = (int) idx;
    int hi = Math.min(lo + 1, count - 1);
    double frac = idx - lo;
    double nanos = samples[lo] * (1 - frac) + samples[hi] * frac;
    return nanos / 1_000.0; // ns → us
  }
}
