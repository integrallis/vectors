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

import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.persist.StudyStore;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class IndexStudyTest {

  @Test
  void sixTrialStudyFindsRecallAboveThreshold(@TempDir Path tmp) {
    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.FixedString("metric", "COSINE"),
                new ParamSpec.FixedString("indexType", "HNSW"),
                new ParamSpec.Discrete<>("m", List.of(8, 16, 32)),
                new ParamSpec.Discrete<>("efConstruction", List.of(64, 128, 200))));

    StudyConfig cfg =
        StudyConfig.builder()
            .searchSpace(space)
            .objectiveWeights(ObjectiveWeights.builder().recallWeight(1.0).ndcgWeight(0.5).build())
            .samplerKind(StudyConfig.SamplerKind.RANDOM)
            .corpusSource(() -> StudyTestFixtures.corpus(42L))
            .qrelsSource(StudyTestFixtures::qrels)
            .queriesSource(StudyTestFixtures::queries)
            .nTrials(6)
            .kForMetrics(10)
            .seed(7L)
            .warmupRounds(1)
            .measurementRounds(2)
            .build();

    StudyStore store = new StudyStore(tmp);
    IndexStudy study = new IndexStudy(cfg, StudyTestFixtures.embedder());
    StudyRunner runner = new StudyRunner("study-recall", study, cfg, store);
    runner.runBlocking();
    runner.close();

    List<TrialResult> trials = store.loadAll("study-recall");
    assertThat(trials).hasSize(6);

    double bestRecall = trials.stream().mapToDouble(TrialResult::recallAtK).max().orElseThrow();
    // Cluster centroids are well-separated; HNSW with any reasonable (m, efConstruction) on this
    // dataset should retrieve all 10 relevant peers in the top-10.
    assertThat(bestRecall).isGreaterThanOrEqualTo(0.7);
  }
}
