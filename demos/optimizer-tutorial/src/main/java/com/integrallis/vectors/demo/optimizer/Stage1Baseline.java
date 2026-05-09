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

import com.integrallis.vectors.db.VectorCollectionBuilder;
import com.integrallis.vectors.optimizer.objective.ObjectiveWeights;
import com.integrallis.vectors.optimizer.space.ParamSpec;
import com.integrallis.vectors.optimizer.space.SearchSpace;
import com.integrallis.vectors.optimizer.space.Trial;
import com.integrallis.vectors.optimizer.study.IndexStudy;
import com.integrallis.vectors.optimizer.study.StudyConfig;
import com.integrallis.vectors.optimizer.study.TrialResult;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stage 1 — Establish a baseline. Builds the corpus with default HNSW parameters ({@code M=16},
 * {@code efConstruction=200}), measures recall@10, NDCG@10 and p95 latency, and persists the
 * snapshot to {@code ~/.vectors/optimizer/tutorial/baseline.json} for later stages to compare
 * against.
 */
public final class Stage1Baseline {

  private Stage1Baseline() {}

  public static void main(String[] args) {
    System.out.println("=== Stage 1 — Baseline (default HNSW: M=16, efConstruction=200) ===");
    TutorialDataset ds = TutorialDataset.loadDefault();
    System.out.printf(
        "Dataset: %s   corpus=%d   queries=%d   dim=%d%n",
        ds.datasetName(), ds.corpus().size(), ds.queries().size(), ds.dimension());

    int defaultM = VectorCollectionBuilder.DEFAULT_HNSW_M;
    int defaultEfC = VectorCollectionBuilder.DEFAULT_HNSW_EF_CONSTRUCTION;

    SearchSpace space =
        new SearchSpace(
            List.of(
                new ParamSpec.FixedString("metric", "EUCLIDEAN"),
                new ParamSpec.FixedString("indexType", "HNSW"),
                new ParamSpec.FixedInt("m", defaultM),
                new ParamSpec.FixedInt("efConstruction", defaultEfC)));

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
            .nTrials(1)
            .kForMetrics(10)
            .seed(42L)
            .warmupRounds(1)
            .measurementRounds(3)
            .build();

    Map<String, Object> params = new LinkedHashMap<>();
    params.put("metric", "EUCLIDEAN");
    params.put("indexType", "HNSW");
    params.put("m", defaultM);
    params.put("efConstruction", defaultEfC);
    Trial baselineTrial = new Trial("baseline", params);

    long t0 = System.nanoTime();
    TrialResult result = new IndexStudy(cfg, ds.embedder()).runOne(baselineTrial);
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;

    TutorialPaths.Snapshot snap =
        new TutorialPaths.Snapshot(
            "baseline",
            ds.datasetName(),
            ds.corpus().size(),
            ds.queries().size(),
            cfg.kForMetrics(),
            defaultM,
            defaultEfC,
            -1,
            result.recallAtK(),
            result.ndcgAtK(),
            result.latencyP50Us(),
            result.latencyP95Us(),
            result.latencyP99Us(),
            result.buildTimeMs(),
            result.objectiveScore());
    TutorialPaths.writeJson(TutorialPaths.baselineFile(), snap);

    System.out.println();
    System.out.println("Baseline");
    System.out.println("┌──────────────────────┬─────────────────────────────────┐");
    System.out.printf("│ Build time           │ %29s │%n", result.buildTimeMs() + " ms");
    System.out.printf(
        "│ Recall@10            │ %29s │%n", String.format("%.4f", result.recallAtK()));
    System.out.printf("│ NDCG@10              │ %29s │%n", String.format("%.4f", result.ndcgAtK()));
    System.out.printf(
        "│ Latency p50 / p95    │ %29s │%n",
        String.format("%.1f µs / %.1f µs", result.latencyP50Us(), result.latencyP95Us()));
    System.out.printf(
        "│ Objective score      │ %29s │%n", String.format("%.4f", result.objectiveScore()));
    System.out.println("└──────────────────────┴─────────────────────────────────┘");
    System.out.printf("Wrote %s   (total wall: %d ms)%n", TutorialPaths.baselineFile(), elapsedMs);
    System.out.println();
    System.out.println("Next: ./gradlew :demos:optimizer-tutorial:stage2BroadSweep");
  }
}
