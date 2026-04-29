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
package com.integrallis.vectors.studio.core.recommender;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.studio.core.projection.ProjectionAlgorithm;
import com.integrallis.vectors.studio.core.projection.ProjectionParams;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class HeuristicRecommenderTest {

  private final HeuristicRecommender r = new HeuristicRecommender();

  @Test
  void sparseDatasetGetsPca() {
    var stats = new DatasetStats(1000, 768, 0.0, null, false, 0.95);
    var rec = r.recommend(stats, 2);
    assertThat(rec.algorithm()).isEqualTo(ProjectionAlgorithm.PCA);
    assertThat(rec.params()).isInstanceOf(ProjectionParams.PcaParams.class);
  }

  @Test
  void dWayMoreThanNGetsPca() {
    var stats = new DatasetStats(50, 1024, 0.0, null, false, 0.0);
    var rec = r.recommend(stats, 2);
    assertThat(rec.algorithm()).isEqualTo(ProjectionAlgorithm.PCA);
  }

  @Test
  void smallNGetsTsne() {
    var stats = new DatasetStats(1000, 32, 8.0, null, false, 0.0);
    var rec = r.recommend(stats, 2);
    assertThat(rec.algorithm()).isEqualTo(ProjectionAlgorithm.TSNE);
    assertThat(rec.params()).isInstanceOf(ProjectionParams.TsneParams.class);
  }

  @Test
  void mediumNGetsUmap() {
    var stats = new DatasetStats(50_000, 128, 16.0, null, false, 0.0);
    var rec = r.recommend(stats, 2);
    assertThat(rec.algorithm()).isEqualTo(ProjectionAlgorithm.UMAP);
  }

  @Test
  void largeNGetsPcaPipeline() {
    var stats = new DatasetStats(1_000_000, 256, 32.0, null, false, 0.0);
    var rec = r.recommend(stats, 2);
    assertThat(rec.algorithm()).isEqualTo(ProjectionAlgorithm.PCA);
    var p = (ProjectionParams.PcaParams) rec.params();
    assertThat(p.components()).isEqualTo(50);
  }
}
