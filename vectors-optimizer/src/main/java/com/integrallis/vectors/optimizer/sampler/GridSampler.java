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

/**
 * Deterministic Cartesian-product enumeration over a finite-cardinality {@link SearchSpace}.
 *
 * <p>Continuous axes ({@link ParamSpec.DoubleRange}) are rejected: callers must wrap the values
 * they want to sweep in {@link ParamSpec.Discrete}. Trial IDs are zero-padded to make sort-by-id
 * align with sort-by-cursor for any given study size.
 */
public final class GridSampler implements ParamSampler {

  private final SearchSpace space;
  private final long total;
  private final long[] strides;
  private final int[] sizes;
  private long cursor;

  public GridSampler(SearchSpace space) {
    this.space = Objects.requireNonNull(space, "space");
    int n = space.axes().size();
    this.sizes = new int[n];
    this.strides = new long[n];
    long product = 1L;
    for (int i = 0; i < n; i++) {
      ParamSpec<?> a = space.axes().get(i);
      if (a instanceof ParamSpec.DoubleRange) {
        throw new IllegalArgumentException(
            "GridSampler cannot enumerate continuous axis "
                + a.name()
                + "; wrap discrete values in ParamSpec.Discrete");
      }
      long card = a.cardinality().orElseThrow();
      sizes[i] = Math.toIntExact(card);
      strides[i] = product;
      product = Math.multiplyExact(product, card);
    }
    this.total = product;
  }

  @Override
  public Trial next(List<? extends ScoredTrial> history) {
    if (cursor >= total) {
      throw new NoMoreTrialsException("GridSampler exhausted after " + total + " trials");
    }
    long c = cursor++;
    Map<String, Object> params = new LinkedHashMap<>();
    for (int i = 0; i < sizes.length; i++) {
      int idx = (int) ((c / strides[i]) % sizes[i]);
      ParamSpec<?> a = space.axes().get(i);
      params.put(a.name(), valueAt(a, idx));
    }
    int width = Math.max(4, Long.toString(total).length());
    String trialId = String.format("grid-%0" + width + "d", c);
    return new Trial(trialId, params);
  }

  private static Object valueAt(ParamSpec<?> a, int idx) {
    return switch (a) {
      case ParamSpec.Categorical c -> c.values().get(idx);
      case ParamSpec.IntRange r -> {
        if (r.logScale()) {
          double t = r.cardinality().getAsLong() == 1 ? 0.0 : (double) idx / (r.max() - r.min());
          double v = Math.exp(Math.log(r.min()) + t * (Math.log(r.max()) - Math.log(r.min())));
          yield (int) Math.round(v);
        }
        yield r.min() + idx;
      }
      case ParamSpec.Discrete<?> d -> d.values().get(idx);
      case ParamSpec.FixedString f -> f.value();
      case ParamSpec.FixedInt f -> f.value();
      case ParamSpec.FixedDouble f -> f.value();
      case ParamSpec.DoubleRange ignored ->
          throw new AssertionError("DoubleRange should have been rejected in constructor");
    };
  }

  /** Total number of trials this sampler will produce. */
  public long total() {
    return total;
  }

  /** Number of trials already produced. */
  public long cursor() {
    return cursor;
  }
}
