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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.semantic.PayloadCodec;
import com.integrallis.vectors.cache.semantic.VectorDbSemanticCache;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.optimizer.sampler.GridSampler;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CacheThresholdStudyTest {

  private static final int DIM = 4;

  /** Build a fresh {@link VectorDbSemanticCache} parameterised by {@code threshold}. */
  private static CacheThresholdStudy.CacheFactory<String> factory() {
    return threshold -> {
      VectorCollection col =
          VectorCollection.builder()
              .dimension(DIM)
              .metric(SimilarityFunction.COSINE)
              .indexType(IndexType.FLAT)
              .build();
      return VectorDbSemanticCache.builder(col, PayloadCodec.identity())
          .threshold(threshold)
          .closeCollectionOnClose(true)
          .build();
    };
  }

  // Seed: one key-A entry at [1,0,0,0].
  private static Map<String, float[]> seeds() {
    Map<String, float[]> s = new LinkedHashMap<>();
    s.put("key-A", new float[] {1f, 0f, 0f, 0f});
    return s;
  }

  /**
   * Probes split into 5 hits (cos ≈ 1.0, 0.99, 0.95, 0.92, 0.90) and 5 misses (cos ≈ 0.6, 0.5,
   * 0.4, 0.3, 0.2). Score = (1+cos)/2 for COSINE, so the analytic optimum threshold lies between
   * the worst hit (score ≈ 0.95) and the best miss (score ≈ 0.80) → ~0.875.
   */
  private static List<LabeledQuery> probes() {
    List<LabeledQuery> p = new ArrayList<>();
    double[] hitCosines = {1.0, 0.99, 0.95, 0.92, 0.90};
    double[] missCosines = {0.60, 0.50, 0.40, 0.30, 0.20};
    for (int i = 0; i < hitCosines.length; i++) {
      p.add(LabeledQuery.cacheProbe("hit-" + i, unitOnX(hitCosines[i]), "key-A"));
    }
    for (int i = 0; i < missCosines.length; i++) {
      p.add(LabeledQuery.cacheProbe("miss-" + i, unitOnX(missCosines[i]), null));
    }
    return p;
  }

  /** Returns a 4-D unit vector whose cosine with [1,0,0,0] is exactly {@code cosX}. */
  private static float[] unitOnX(double cosX) {
    double sin = Math.sqrt(Math.max(0.0, 1.0 - cosX * cosX));
    return new float[] {(float) cosX, (float) sin, 0f, 0f};
  }

  @Test
  void sixtyPointGridFindsBestThreshold() {
    List<Double> grid =
        IntStream.range(0, 60)
            .mapToDouble(i -> 0.01 + (0.99 - 0.01) * i / 59.0)
            .boxed()
            .toList();
    SearchSpace space =
        new SearchSpace(List.of(new ParamSpec.Discrete<>("threshold", grid)));
    GridSampler sampler = new GridSampler(space);
    CacheThresholdStudy<String> study =
        new CacheThresholdStudy<>(factory(), seeds(), probes(), "value");

    double bestScore = Double.NEGATIVE_INFINITY;
    double bestThreshold = Double.NaN;
    for (int i = 0; i < 60; i++) {
      Trial trial = sampler.next(List.of());
      TrialResult tr = study.runOne(trial);
      // Cache scoring: score = (1+cos)/2 → in [0.5, 1.0] for non-negative cosines we tested.
      // Choose the highest-accuracy threshold; ties broken by a smaller threshold.
      if (tr.objectiveScore() > bestScore + 1e-9) {
        bestScore = tr.objectiveScore();
        bestThreshold = (Double) trial.params().get("threshold");
      }
    }
    assertThat(bestScore).as("best accuracy across grid").isEqualTo(1.0);
    // Analytic optimum: between the worst-hit score (≈0.95) and best-miss score (≈0.80),
    // so any threshold in (0.80, 0.95] gives perfect accuracy. The mid-range optimum is ~0.875.
    assertThat(bestThreshold).isBetween(0.80, 0.95);
  }
}
