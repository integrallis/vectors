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
package com.integrallis.vectors.optimizer.sampler;

import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.ScoredTrial;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Univariate Tree-structured Parzen Estimator. Independent per-axis Parzen densities {@code l(x)}
 * (good observations) and {@code g(x)} (bad observations); proposals maximize {@code log l(x) - log
 * g(x)} per axis. References: Bergstra et al. 2011, Watanabe 2023 (arXiv:2304.11127v3), Optuna v3
 * default {@code TPESampler} hyperparameters.
 *
 * <p>Multivariate joint TPE is a v2 follow-up; this implementation samples each axis independently.
 */
public final class TpeSampler implements ParamSampler {

  /** Defaults match Optuna v3's {@code hyperopt_parameters()}. */
  public record Hyperparameters(
      int nStartupTrials, int nEiCandidates, double gamma, double priorWeight) {

    public Hyperparameters {
      if (nStartupTrials < 0) throw new IllegalArgumentException("nStartupTrials >= 0");
      if (nEiCandidates < 1) throw new IllegalArgumentException("nEiCandidates >= 1");
      if (!(gamma > 0.0 && gamma < 1.0)) throw new IllegalArgumentException("0 < gamma < 1");
      if (priorWeight < 0.0) throw new IllegalArgumentException("priorWeight >= 0");
    }

    public static Hyperparameters defaults() {
      return new Hyperparameters(10, 24, 0.25, 1.0);
    }
  }

  private final SearchSpace space;
  private final Hyperparameters hp;
  private final RandomGenerator rng;
  private final RandomSampler startup;
  private long counter;

  public TpeSampler(SearchSpace space, long seed, Hyperparameters hp) {
    this.space = Objects.requireNonNull(space, "space");
    this.hp = Objects.requireNonNull(hp, "hp");
    var factory = RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom");
    this.rng = factory.create(seed);
    // Seed the inner random sampler with a derived seed so its RNG is independent of the
    // candidate-sampling RNG: this lets `delegatesToRandomBeforeStartup` compare TPE against a
    // RandomSampler with the same seed, since under TPE-startup we use the SAME inner sampler.
    this.startup = new RandomSampler(space, seed);
  }

  public TpeSampler(SearchSpace space, long seed) {
    this(space, seed, Hyperparameters.defaults());
  }

  @Override
  public Trial next(List<? extends ScoredTrial> history) {
    if (history.size() < hp.nStartupTrials()) {
      return startup.next(history);
    }

    List<ScoredTrial> sorted = new ArrayList<>(history);
    sorted.sort(Comparator.comparingDouble(ScoredTrial::objectiveScore).reversed());
    int nGood = Math.max(1, (int) Math.round(hp.gamma() * sorted.size()));
    nGood = Math.min(nGood, sorted.size() - 1);
    List<ScoredTrial> good = sorted.subList(0, nGood);
    List<ScoredTrial> bad = sorted.subList(nGood, sorted.size());

    Map<String, Object> chosen = new LinkedHashMap<>();
    for (ParamSpec<?> a : space.axes()) {
      chosen.put(a.name(), proposeAxis(a, good, bad));
    }
    return new Trial("tpe-" + Long.toString(counter++), chosen);
  }

  // Per-axis proposal: sample nEiCandidates from l, score with log l - log g, return argmax.
  private Object proposeAxis(ParamSpec<?> axis, List<ScoredTrial> good, List<ScoredTrial> bad) {
    if (axis instanceof ParamSpec.FixedString f) return f.value();
    if (axis instanceof ParamSpec.FixedInt f) return f.value();
    if (axis instanceof ParamSpec.FixedDouble f) return f.value();

    if (axis instanceof ParamSpec.Categorical || axis instanceof ParamSpec.Discrete<?>) {
      return proposeCategorical(axis, good, bad);
    }
    return proposeContinuous(axis, good, bad);
  }

  private Object proposeCategorical(
      ParamSpec<?> axis, List<ScoredTrial> good, List<ScoredTrial> bad) {
    List<?> values =
        (axis instanceof ParamSpec.Categorical c)
            ? c.values()
            : ((ParamSpec.Discrete<?>) axis).values();
    double[] lPmf = categoricalPmf(axis, good, values);
    double[] gPmf = categoricalPmf(axis, bad, values);

    Object best = values.get(0);
    double bestEi = Double.NEGATIVE_INFINITY;
    for (int n = 0; n < hp.nEiCandidates(); n++) {
      int idx = sampleCategorical(lPmf);
      double ei =
          Math.log(Math.max(lPmf[idx], Double.MIN_NORMAL))
              - Math.log(Math.max(gPmf[idx], Double.MIN_NORMAL));
      if (ei > bestEi) {
        bestEi = ei;
        best = values.get(idx);
      }
    }
    return best;
  }

  private double[] categoricalPmf(ParamSpec<?> axis, List<ScoredTrial> trials, List<?> values) {
    int k = values.size();
    double[] counts = new double[k];
    for (ScoredTrial t : trials) {
      Object v = t.trial().params().get(axis.name());
      int idx = values.indexOf(v);
      if (idx >= 0) counts[idx] += 1.0;
    }
    double total = trials.size() + hp.priorWeight() * k;
    double[] pmf = new double[k];
    for (int i = 0; i < k; i++) pmf[i] = (counts[i] + hp.priorWeight()) / total;
    return pmf;
  }

  private int sampleCategorical(double[] pmf) {
    double u = rng.nextDouble();
    double cumulative = 0.0;
    for (int i = 0; i < pmf.length; i++) {
      cumulative += pmf[i];
      if (u <= cumulative) return i;
    }
    return pmf.length - 1;
  }

  private Object proposeContinuous(
      ParamSpec<?> axis, List<ScoredTrial> good, List<ScoredTrial> bad) {
    Bounds b = boundsOf(axis);
    double[] goodObs = extractContinuous(axis, good, b);
    double[] badObs = extractContinuous(axis, bad, b);

    double bestX = Double.NaN;
    double bestEi = Double.NEGATIVE_INFINITY;
    for (int n = 0; n < hp.nEiCandidates(); n++) {
      double x = sampleParzen(goodObs, b);
      double lpdf = parzenPdf(x, goodObs, b);
      double gpdf = parzenPdf(x, badObs, b);
      double ei =
          Math.log(Math.max(lpdf, Double.MIN_NORMAL)) - Math.log(Math.max(gpdf, Double.MIN_NORMAL));
      if (ei > bestEi || Double.isNaN(bestX)) {
        bestEi = ei;
        bestX = x;
      }
    }
    return materializeContinuous(axis, bestX, b);
  }

  // --- continuous helpers ---

  private record Bounds(double lo, double hi, boolean log) {}

  private static Bounds boundsOf(ParamSpec<?> axis) {
    return switch (axis) {
      case ParamSpec.IntRange r ->
          r.logScale()
              ? new Bounds(Math.log(r.min()), Math.log(r.max()), true)
              : new Bounds(r.min(), r.max(), false);
      case ParamSpec.DoubleRange r ->
          r.logScale()
              ? new Bounds(Math.log(r.min()), Math.log(r.max()), true)
              : new Bounds(r.min(), r.max(), false);
      default -> throw new IllegalArgumentException("Not a continuous axis: " + axis.name());
    };
  }

  private static double[] extractContinuous(ParamSpec<?> axis, List<ScoredTrial> trials, Bounds b) {
    double[] out = new double[trials.size()];
    int n = 0;
    for (ScoredTrial t : trials) {
      Object v = t.trial().params().get(axis.name());
      if (v == null) continue;
      double x = ((Number) v).doubleValue();
      out[n++] = b.log() ? Math.log(x) : x;
    }
    return n == out.length ? out : java.util.Arrays.copyOf(out, n);
  }

  private double sampleParzen(double[] obs, Bounds b) {
    double total = obs.length + hp.priorWeight();
    double u = rng.nextDouble() * total;
    if (u < obs.length) {
      int i = Math.min(obs.length - 1, (int) u);
      double sigma = sigmaAt(obs, i, b);
      double x;
      do {
        x = obs[i] + rng.nextGaussian() * sigma;
      } while (x < b.lo() || x > b.hi());
      return x;
    }
    return b.lo() + rng.nextDouble() * (b.hi() - b.lo());
  }

  private double parzenPdf(double x, double[] obs, Bounds b) {
    double total = obs.length + hp.priorWeight();
    double pdf = 0.0;
    double range = b.hi() - b.lo();
    for (int i = 0; i < obs.length; i++) {
      double sigma = sigmaAt(obs, i, b);
      double z = (x - obs[i]) / sigma;
      pdf += (1.0 / total) * Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2.0 * Math.PI));
    }
    pdf += (hp.priorWeight() / total) * (1.0 / Math.max(range, Double.MIN_NORMAL));
    return pdf;
  }

  private double sigmaAt(double[] obs, int i, Bounds b) {
    double range = b.hi() - b.lo();
    if (obs.length == 0) return range;
    if (obs.length == 1) return range / 2.0;
    double floor = range / Math.max(8.0, Math.sqrt(obs.length + 1.0));
    double[] sorted = obs.clone();
    java.util.Arrays.sort(sorted);
    int idx = java.util.Arrays.binarySearch(sorted, obs[i]);
    if (idx < 0) idx = -idx - 1;
    // Boundary points mirror the only available neighbor distance instead of using the full range,
    // so leftmost/rightmost kernels do not become absurdly wide on small samples.
    double left = idx > 0 ? sorted[idx] - sorted[idx - 1] : Double.NaN;
    double right = idx < sorted.length - 1 ? sorted[idx + 1] - sorted[idx] : Double.NaN;
    double sigma;
    if (Double.isNaN(left)) sigma = right;
    else if (Double.isNaN(right)) sigma = left;
    else sigma = Math.max(left, right);
    return Math.max(floor, sigma);
  }

  private static Object materializeContinuous(ParamSpec<?> axis, double x, Bounds b) {
    double real = b.log() ? Math.exp(x) : x;
    return switch (axis) {
      case ParamSpec.IntRange r -> {
        int v = (int) Math.round(real);
        yield Math.max(r.min(), Math.min(r.max(), v));
      }
      case ParamSpec.DoubleRange r -> Math.max(r.min(), Math.min(r.max(), real));
      default -> throw new IllegalStateException("Not a continuous axis: " + axis.name());
    };
  }
}
