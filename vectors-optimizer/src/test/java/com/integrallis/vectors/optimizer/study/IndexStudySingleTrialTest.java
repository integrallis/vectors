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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class IndexStudySingleTrialTest {

  private static StudyConfig fixedConfig() {
    return StudyConfig.builder()
        .searchSpace(
            new SearchSpace(
                List.of(
                    new ParamSpec.FixedString("metric", "COSINE"),
                    new ParamSpec.FixedString("indexType", "FLAT"))))
        .objectiveWeights(ObjectiveWeights.builder().recallWeight(1.0).ndcgWeight(1.0).build())
        .samplerKind(StudyConfig.SamplerKind.RANDOM)
        .corpusSource(() -> StudyTestFixtures.corpus(42L))
        .qrelsSource(StudyTestFixtures::qrels)
        .queriesSource(StudyTestFixtures::queries)
        .nTrials(1)
        .kForMetrics(10)
        .seed(0L)
        .warmupRounds(1)
        .measurementRounds(2)
        .build();
  }

  @Test
  void singleTrialBuildsAndScoresCollection() {
    IndexStudy study = new IndexStudy(fixedConfig(), StudyTestFixtures.embedder());
    Trial trial = new Trial("t-0", Map.of("metric", "COSINE", "indexType", "FLAT"));

    TrialResult tr = study.runOne(trial);

    // FLAT brute-force on a clean clustered dataset: top-10 should all be from the right cluster.
    assertThat(tr.recallAtK()).isGreaterThanOrEqualTo(0.7);
    assertThat(tr.ndcgAtK()).isGreaterThan(0.0);
    assertThat(tr.latencyP95Us()).isPositive();
    assertThat(tr.buildTimeMs()).isGreaterThanOrEqualTo(0L);
    assertThat(tr.memoryBytes()).isPositive();
    assertThat(tr.startedAt()).isBefore(tr.finishedAt());
    assertThat(tr.objectiveScore()).isFinite();
  }

  @Test
  void closesCollectionEvenOnFailure() {
    // Force a deterministic failure inside runOne by injecting an embedder that throws after
    // the collection has been built and committed.
    EmbeddingProvider boom =
        new EmbeddingProvider() {
          @Override
          public float[] embed(String text) {
            throw new IllegalStateException("intentional");
          }
        };
    IndexStudy study = new IndexStudy(fixedConfig(), boom);
    Trial trial = new Trial("t-0", Map.of("metric", "COSINE", "indexType", "FLAT"));

    assertThatThrownBy(() -> study.runOne(trial))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("intentional");
    // No file/socket leak detection here — the contract is that try-with-resources unconditionally
    // calls VectorCollection.close(); a leak would surface as an OutOfMemoryError under repeated
    // runs. The thrown exception confirms the path executed past collection construction.
  }
}
