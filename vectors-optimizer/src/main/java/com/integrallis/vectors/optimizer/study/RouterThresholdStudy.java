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
import com.integrallis.vectors.optimizer.space.Trial;
import com.integrallis.vectors.router.RouteMatch;
import com.integrallis.vectors.router.SemanticRouter;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Sweeps per-route similarity thresholds for a {@link SemanticRouter}. Each trial constructs a
 * fresh router from a caller-supplied {@link RouterFactory} (the router is immutable, so threshold
 * changes mean a new instance per trial), then evaluates micro-averaged accuracy across the
 * labelled probe set.
 *
 * <p>For single-label classification micro-precision == micro-recall == accuracy; the resulting
 * scalar is reported as {@code recall/ndcg/precision/f1} for downstream consumption by {@link
 * com.integrallis.vectors.optimizer.objective.Objective}.
 */
public final class RouterThresholdStudy {

  /** Constructs a {@link SemanticRouter} from per-route distance thresholds. */
  @FunctionalInterface
  public interface RouterFactory {
    SemanticRouter create(Map<String, Double> thresholdsByRoute);
  }

  /**
   * Trial parameter prefix; an axis named {@code "threshold_<route>"} carries that route's
   * threshold.
   */
  public static final String THRESHOLD_PREFIX = "threshold_";

  private final RouterFactory factory;
  private final List<LabeledQuery> labeled;

  public RouterThresholdStudy(RouterFactory factory, List<LabeledQuery> labeled) {
    this.factory = Objects.requireNonNull(factory, "factory");
    Objects.requireNonNull(labeled, "labeled");
    if (labeled.isEmpty()) throw new IllegalArgumentException("labeled must be non-empty");
    this.labeled = List.copyOf(labeled);
  }

  /**
   * Executes one trial: build the router with {@code trial}'s thresholds and score the probe set.
   */
  public TrialResult runOne(Trial trial) {
    Objects.requireNonNull(trial, "trial");
    Instant startedAt = Instant.now();
    long buildStart = System.nanoTime();
    SemanticRouter router = factory.create(extractThresholds(trial));
    long buildTimeMs = (System.nanoTime() - buildStart) / 1_000_000L;

    LatencyCollector lc = new LatencyCollector(labeled.size());
    int correct = 0;
    for (LabeledQuery q : labeled) {
      long t0 = System.nanoTime();
      RouteMatch match = router.route(q.text());
      lc.record(System.nanoTime() - t0);
      String predicted = match == null ? null : match.getName();
      if (Objects.equals(predicted, q.expectedLabel())) correct++;
    }
    lc.compute();
    double accuracy = (double) correct / labeled.size();
    return new TrialResult(
        trial,
        startedAt,
        Instant.now(),
        accuracy,
        accuracy,
        accuracy,
        accuracy,
        0.0,
        lc.p50Us(),
        lc.p95Us(),
        lc.p99Us(),
        buildTimeMs,
        0L,
        accuracy);
  }

  private static Map<String, Double> extractThresholds(Trial trial) {
    Map<String, Double> out = new LinkedHashMap<>();
    for (var e : trial.params().entrySet()) {
      String key = e.getKey();
      if (key.startsWith(THRESHOLD_PREFIX)) {
        out.put(key.substring(THRESHOLD_PREFIX.length()), ((Number) e.getValue()).doubleValue());
      }
    }
    if (out.isEmpty()) {
      throw new IllegalArgumentException(
          "Trial parameters contain no axes named '"
              + THRESHOLD_PREFIX
              + "<route>'; got: "
              + trial.params().keySet());
    }
    return out;
  }
}
