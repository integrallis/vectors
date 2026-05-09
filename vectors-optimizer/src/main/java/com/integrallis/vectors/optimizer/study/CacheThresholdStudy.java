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
package com.integrallis.vectors.optimizer.study;

import com.integrallis.vectors.bench.report.LatencyCollector;
import com.integrallis.vectors.cache.SemanticCache;
import com.integrallis.vectors.optimizer.space.Trial;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Sweeps a single similarity threshold for a {@link SemanticCache}. Each trial creates a fresh
 * cache via the supplied {@link CacheFactory}, seeds it with the caller-provided
 * {@code (key, embedding)} pairs, then probes every labelled query and computes accuracy
 * (per-query: a hit returning {@code expectedLabel} is correct; a miss is correct iff
 * {@code expectedLabel == null}).
 *
 * <p>Pair with {@link com.integrallis.vectors.optimizer.sampler.GridSampler} on a single
 * {@code DoubleRange("threshold", lo, hi)} axis, or wrap a discrete grid via
 * {@link com.integrallis.vectors.optimizer.space.ParamSpec.Discrete}.
 */
public final class CacheThresholdStudy<V> {

  /** Constructs a {@link SemanticCache} parameterised by the trial threshold. */
  @FunctionalInterface
  public interface CacheFactory<V> {
    SemanticCache<V> create(double threshold);
  }

  /** Default trial-axis name carrying the threshold value. */
  public static final String THRESHOLD_AXIS = "threshold";

  private final CacheFactory<V> factory;
  private final Map<String, float[]> seeds;
  private final List<LabeledQuery> probes;
  private final V seedValue;

  public CacheThresholdStudy(
      CacheFactory<V> factory,
      Map<String, float[]> seeds,
      List<LabeledQuery> probes,
      V seedValue) {
    this.factory = Objects.requireNonNull(factory, "factory");
    Objects.requireNonNull(seeds, "seeds");
    Objects.requireNonNull(probes, "probes");
    if (probes.isEmpty()) throw new IllegalArgumentException("probes must be non-empty");
    this.seeds = Map.copyOf(seeds);
    this.probes = List.copyOf(probes);
    this.seedValue = seedValue;
  }

  /** Executes one trial: build the cache, seed it, probe it, score accuracy. */
  public TrialResult runOne(Trial trial) {
    Objects.requireNonNull(trial, "trial");
    Object raw = trial.params().get(THRESHOLD_AXIS);
    if (raw == null) {
      throw new IllegalArgumentException(
          "Trial is missing the required '" + THRESHOLD_AXIS + "' axis; got: " + trial.params().keySet());
    }
    double threshold = ((Number) raw).doubleValue();
    Instant startedAt = Instant.now();
    long buildStart = System.nanoTime();
    LatencyCollector lc = new LatencyCollector(probes.size());
    int correct = 0;

    try (SemanticCache<V> cache = factory.create(threshold)) {
      for (var e : seeds.entrySet()) cache.put(e.getKey(), e.getValue(), seedValue);
      long buildTimeMs = (System.nanoTime() - buildStart) / 1_000_000L;

      for (LabeledQuery q : probes) {
        if (q.embedding() == null) {
          throw new IllegalStateException(
              "CacheThresholdStudy probes require pre-computed embeddings; missing for '"
                  + q.text()
                  + "'");
        }
        long t0 = System.nanoTime();
        Optional<SemanticCache.Hit<V>> hit = cache.lookup(q.embedding());
        lc.record(System.nanoTime() - t0);
        // A "hit" matches the expected label by convention: the seed was put under that key,
        // and lookup returning the same key counts as success. Since lookup returns Hit<V>
        // (value only), we infer correctness via threshold semantics: if expected != null we
        // expect a hit; if expected == null we expect a miss.
        boolean expectsHit = q.expectedLabel() != null;
        if (hit.isPresent() == expectsHit) correct++;
      }
      lc.compute();
      double accuracy = (double) correct / probes.size();
      return new TrialResult(
          trial,
          startedAt,
          Instant.now(),
          accuracy, accuracy, accuracy, accuracy, 0.0,
          lc.p50Us(), lc.p95Us(), lc.p99Us(),
          buildTimeMs, 0L,
          accuracy);
    }
  }
}
