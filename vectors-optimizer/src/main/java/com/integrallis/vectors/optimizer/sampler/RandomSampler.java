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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Per-axis uniform sampler. Continuous axes with {@code logScale=true} are sampled in log-space and
 * exponentiated; integer ranges with {@code logScale} pick a real value in log-space and round.
 */
public final class RandomSampler implements ParamSampler {

  private final SearchSpace space;
  private final RandomGenerator rng;
  private long counter;

  public RandomSampler(SearchSpace space, long seed) {
    this(space, seed, RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom"));
  }

  public RandomSampler(
      SearchSpace space, long seed, RandomGeneratorFactory<RandomGenerator> factory) {
    this.space = Objects.requireNonNull(space, "space");
    Objects.requireNonNull(factory, "factory");
    this.rng = factory.create(seed);
  }

  @Override
  public Trial next(List<? extends ScoredTrial> history) {
    Map<String, Object> params = new LinkedHashMap<>();
    for (ParamSpec<?> a : space.axes()) {
      params.put(a.name(), sample(a));
    }
    return new Trial("random-" + Long.toString(counter++), params);
  }

  /** Samples a single axis without consuming the joint trial counter. Package-private for TPE. */
  Object sample(ParamSpec<?> a) {
    return switch (a) {
      case ParamSpec.Categorical c -> c.values().get(rng.nextInt(c.values().size()));
      case ParamSpec.IntRange r -> {
        if (r.logScale()) {
          double lo = Math.log(r.min());
          double hi = Math.log(r.max());
          double v = Math.exp(lo + rng.nextDouble() * (hi - lo));
          yield clampInt((int) Math.round(v), r.min(), r.max());
        }
        yield r.min() + rng.nextInt(r.max() - r.min() + 1);
      }
      case ParamSpec.DoubleRange r -> {
        if (r.logScale()) {
          double lo = Math.log(r.min());
          double hi = Math.log(r.max());
          yield Math.exp(lo + rng.nextDouble() * (hi - lo));
        }
        yield r.min() + rng.nextDouble() * (r.max() - r.min());
      }
      case ParamSpec.Discrete<?> d -> d.values().get(rng.nextInt(d.values().size()));
      case ParamSpec.FixedString f -> f.value();
      case ParamSpec.FixedInt f -> f.value();
      case ParamSpec.FixedDouble f -> f.value();
    };
  }

  private static int clampInt(int v, int lo, int hi) {
    return Math.max(lo, Math.min(hi, v));
  }
}
