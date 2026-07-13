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
  // Per-axis enumerated distinct values, in grid order. Precomputing (instead of computing valueAt
  // from cardinality on the fly) is what makes a log-scale IntRange correct: rounding log-interpolated
  // integers collapses many linear indices onto the same value, so the real grid is the DISTINCT set,
  // not max-min+1 points. sizes[i]/total are derived from these lists so total() never over-reports.
  private final List<List<Object>> axisValues;
  private long cursor;

  public GridSampler(SearchSpace space) {
    this.space = Objects.requireNonNull(space, "space");
    int n = space.axes().size();
    this.sizes = new int[n];
    this.strides = new long[n];
    this.axisValues = new ArrayList<>(n);
    long product = 1L;
    for (int i = 0; i < n; i++) {
      ParamSpec<?> a = space.axes().get(i);
      if (a instanceof ParamSpec.DoubleRange) {
        throw new IllegalArgumentException(
            "GridSampler cannot enumerate continuous axis "
                + a.name()
                + "; wrap discrete values in ParamSpec.Discrete");
      }
      List<Object> vals = enumerate(a);
      axisValues.add(vals);
      sizes[i] = vals.size();
      strides[i] = product;
      product = Math.multiplyExact(product, vals.size());
    }
    this.total = product;
  }

  /** Enumerates an axis's distinct grid values in order (log-scale IntRange values are deduped). */
  private static List<Object> enumerate(ParamSpec<?> a) {
    List<Object> vals = new ArrayList<>();
    switch (a) {
      case ParamSpec.Categorical c -> vals.addAll(c.values());
      case ParamSpec.Discrete<?> d -> vals.addAll(d.values());
      case ParamSpec.FixedString f -> vals.add(f.value());
      case ParamSpec.FixedInt f -> vals.add(f.value());
      case ParamSpec.FixedDouble f -> vals.add(f.value());
      case ParamSpec.IntRange r -> {
        if (r.logScale()) {
          long steps = (long) r.max() - r.min();
          double logMin = Math.log(r.min());
          double logMax = Math.log(r.max());
          int prev = Integer.MIN_VALUE;
          for (long idx = 0; idx <= steps; idx++) {
            double t = steps == 0 ? 0.0 : (double) idx / steps;
            int v = (int) Math.round(Math.exp(logMin + t * (logMax - logMin)));
            // Log interpolation is monotonic non-decreasing, so a consecutive-dedup is a full dedup.
            if (v != prev) {
              vals.add(v);
              prev = v;
            }
          }
        } else {
          for (int v = r.min(); v <= r.max(); v++) {
            vals.add(v);
          }
        }
      }
      case ParamSpec.DoubleRange ignored ->
          throw new AssertionError("DoubleRange should have been rejected in constructor");
    }
    return vals;
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
      params.put(space.axes().get(i).name(), axisValues.get(i).get(idx));
    }
    int width = Math.max(4, Long.toString(total).length());
    String trialId = String.format("grid-%0" + width + "d", c);
    return new Trial(trialId, params);
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
