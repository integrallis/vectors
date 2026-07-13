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
package com.integrallis.vectors.ivf;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.core.SimilarityFunction;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Regression for the COSINE zero-norm defect: a zero-norm stored vector (or a zero query) made the
 * bulk-scan divide by zero → NaN, which slipped past the {@code score < minScore} filter (NaN &lt; x
 * is false) and poisoned {@code TopKHeap} ordering. Every returned score must now be finite.
 */
@Tag("unit")
class IvfCosineZeroNormTest {

  @Test
  void zeroStoredVectorUnderCosineYieldsFiniteScores() {
    float[][] data = {
      {1f, 0f, 0f, 0f},
      {0f, 1f, 0f, 0f},
      {0f, 0f, 1f, 0f},
      {0f, 0f, 0f, 0f}, // zero vector — no direction, norm 0
      {1f, 1f, 0f, 0f},
      {0.5f, 0.5f, 0.5f, 0.5f},
    };
    IvfIndex idx =
        IvfIndex.build(
            data, null, SimilarityFunction.COSINE, new IvfBuildParams(2, 10, 0f, false, 42L, 0));

    // nprobe covers both clusters and k covers all rows, so the zero vector is definitely scored
    // and (pre-fix) its NaN score would be admitted to the result set.
    IvfSearchResult r =
        idx.search(
            new IvfSearchRequest(new float[] {1f, 1f, 0f, 0f}, data.length, 2, 0f, -Float.MAX_VALUE));

    assertThat(r.hits()).isNotEmpty();
    for (IvfHit h : r.hits()) {
      assertThat(Float.isFinite(h.score()))
          .as("hit ordinal %d score must be finite, got %s", h.ordinal(), h.score())
          .isTrue();
    }
  }

  @Test
  void zeroQueryVectorUnderCosineYieldsFiniteScores() {
    float[][] data = {{1f, 0f}, {0f, 1f}, {1f, 1f}};
    IvfIndex idx =
        IvfIndex.build(
            data, null, SimilarityFunction.COSINE, new IvfBuildParams(1, 10, 0f, false, 1L, 0));

    IvfSearchResult r =
        idx.search(new IvfSearchRequest(new float[] {0f, 0f}, 3, 1, 0f, -Float.MAX_VALUE));

    for (IvfHit h : r.hits()) {
      assertThat(Float.isFinite(h.score())).as("score must be finite").isTrue();
    }
  }
}
