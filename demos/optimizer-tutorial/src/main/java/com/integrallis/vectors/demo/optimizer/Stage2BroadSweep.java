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
package com.integrallis.vectors.demo.optimizer;

import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.persist.StudyStore;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.study.IndexStudy;
import com.integrallis.vectors.optimizer.study.StudyConfig;
import com.integrallis.vectors.optimizer.study.StudyRunner;
import com.integrallis.vectors.optimizer.study.TrialResult;
import java.util.Comparator;
import java.util.List;

/**
 * Stage 2 — Broad random sweep. Runs a 20-trial Random search across {@code m ∈ [8, 64]} and {@code
 * efConstruction ∈ [40, 300]} to find the rough region where good configurations live, with minimal
 * assumptions about the surface. Results are persisted via {@link StudyStore} so Stage 3 (TPE) can
 * warm-start from the same history if desired.
 */
public final class Stage2BroadSweep {

  private static final String STUDY_ID = "tutorial-stage2-random";

  private Stage2BroadSweep() {}

  public static void main(String[] args) {
    System.out.println("=== Stage 2 — Broad random sweep (20 trials, m & efConstruction) ===");
    TutorialDataset ds = TutorialDataset.loadDefault();
    System.out.printf(
        "Dataset: %s   corpus=%d   queries=%d   dim=%d%n",
        ds.datasetName(), ds.corpus().size(), ds.queries().size(), ds.dimension());

    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.FixedString("metric", "EUCLIDEAN"),
                new ParamSpec.FixedString("indexType", "HNSW"),
                new ParamSpec.IntRange("m", 8, 64, true),
                new ParamSpec.IntRange("efConstruction", 40, 300, true)));

    StudyConfig cfg =
        StudyConfig.builder()
            .searchSpace(space)
            .objectiveWeights(
                ObjectiveWeights.builder()
                    .recallWeight(1.0)
                    .ndcgWeight(0.5)
                    .latencyP95Weight(0.5)
                    .latencyP95ReferenceUs(500.0)
                    .buildTimeWeight(0.3)
                    .buildTimeReferenceMs(15_000.0)
                    .build())
            .samplerKind(StudyConfig.SamplerKind.RANDOM)
            .corpusSource(ds::corpus)
            .qrelsSource(ds::qrels)
            .queriesSource(ds::queries)
            .nTrials(20)
            .kForMetrics(10)
            .seed(42L)
            .warmupRounds(1)
            .measurementRounds(3)
            .build();

    StudyStore store = StudyStore.defaultRoot();
    long t0 = System.nanoTime();
    try (StudyRunner runner =
        new StudyRunner(STUDY_ID, new IndexStudy(cfg, ds.embedder()), cfg, store)) {
      runner.runBlocking();
    }
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;

    List<TrialResult> trials = store.loadAll(STUDY_ID);
    trials.sort(Comparator.comparingDouble(TrialResult::objectiveScore).reversed());
    int top = Math.min(5, trials.size());
    System.out.println();
    System.out.println("Top " + top + " trials by objective:");
    System.out.printf(
        "  %-3s  %-6s  %-15s  %-9s  %-9s  %-12s  %-9s%n",
        "#", "m", "efConstruction", "recall@10", "ndcg@10", "p95 latency", "score");
    for (int i = 0; i < top; i++) {
      TrialResult t = trials.get(i);
      Object m = t.trial().params().get("m");
      Object efc = t.trial().params().get("efConstruction");
      System.out.printf(
          "  %-3d  %-6s  %-15s  %-9.4f  %-9.4f  %-9.1f µs  %-9.4f%n",
          i + 1, m, efc, t.recallAtK(), t.ndcgAtK(), t.latencyP95Us(), t.objectiveScore());
    }
    System.out.printf(
        "%n%d trials persisted to ~/.vectors/optimizer/studies/%s.jsonl  (wall: %d ms)%n",
        trials.size(), STUDY_ID, elapsedMs);
    System.out.println();
    System.out.println("Next: ./gradlew :demos:optimizer-tutorial:stage3Tpe");
  }
}
