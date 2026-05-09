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
 * Stage 3 — Bayesian fine-tuning with TPE. Runs a 40-trial Tree-structured Parzen Estimator study
 * over {@code m}, {@code efConstruction}, and {@code efSearch}, then prints a side-by-side
 * comparison versus the baseline written by Stage 1. The best configuration is persisted to {@code
 * ~/.vectors/optimizer/tutorial/best.json}.
 */
public final class Stage3TpeOptimization {

  private static final String STUDY_ID = "tutorial-stage3-tpe";

  private Stage3TpeOptimization() {}

  public static void main(String[] args) {
    System.out.println(
        "=== Stage 3 — TPE optimization (40 trials, m & efConstruction & efSearch) ===");
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
                new ParamSpec.IntRange("efConstruction", 40, 300, true),
                new ParamSpec.IntRange("efSearch", 16, 200, true)));

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
            .samplerKind(StudyConfig.SamplerKind.TPE)
            .corpusSource(ds::corpus)
            .qrelsSource(ds::qrels)
            .queriesSource(ds::queries)
            .nTrials(40)
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
    TrialResult best = trials.get(0);
    int bestM = ((Number) best.trial().params().get("m")).intValue();
    int bestEfC = ((Number) best.trial().params().get("efConstruction")).intValue();
    int bestEfS = ((Number) best.trial().params().get("efSearch")).intValue();

    TutorialPaths.Snapshot bestSnap =
        new TutorialPaths.Snapshot(
            "tpe-best",
            ds.datasetName(),
            ds.corpus().size(),
            ds.queries().size(),
            cfg.kForMetrics(),
            bestM,
            bestEfC,
            bestEfS,
            best.recallAtK(),
            best.ndcgAtK(),
            best.latencyP50Us(),
            best.latencyP95Us(),
            best.latencyP99Us(),
            best.buildTimeMs(),
            best.objectiveScore());
    TutorialPaths.writeJson(TutorialPaths.bestFile(), bestSnap);

    TutorialPaths.Snapshot baseline = TutorialPaths.readSnapshot(TutorialPaths.baselineFile());
    System.out.println();
    if (baseline == null) {
      System.out.println(
          "(No baseline snapshot found — run :stage1Baseline first to see a comparison.)");
    } else {
      printComparison(baseline, bestSnap);
    }
    System.out.printf(
        "%n40 trials persisted to ~/.vectors/optimizer/studies/%s.jsonl  (wall: %d ms)%n",
        STUDY_ID, elapsedMs);
    System.out.printf("Best snapshot written to %s%n", TutorialPaths.bestFile());
    System.out.println();
    System.out.println("Next: ./gradlew :demos:optimizer-tutorial:stage4Threshold");
  }

  private static void printComparison(TutorialPaths.Snapshot before, TutorialPaths.Snapshot after) {
    System.out.println("Before vs. after");
    System.out.println("┌─────────────────────┬───────────────┬───────────────┬───────────────┐");
    System.out.println("│ Metric              │      Baseline │      TPE-best │             Δ │");
    System.out.println("├─────────────────────┼───────────────┼───────────────┼───────────────┤");
    row("recall@10", before.recallAtK(), after.recallAtK(), "%.4f", true);
    row("ndcg@10", before.ndcgAtK(), after.ndcgAtK(), "%.4f", true);
    row("p95 latency (µs)", before.p95LatencyUs(), after.p95LatencyUs(), "%.1f", false);
    row("build time (ms)", before.buildTimeMs(), after.buildTimeMs(), "%.0f", false);
    row("objective score", before.objectiveScore(), after.objectiveScore(), "%.4f", true);
    System.out.println("└─────────────────────┴───────────────┴───────────────┴───────────────┘");
    System.out.printf(
        "Best params: m=%d  efConstruction=%d  efSearch=%d%n",
        after.m(), after.efConstruction(), after.efSearch());
  }

  private static void row(String label, double a, double b, String fmt, boolean higherIsBetter) {
    double delta = b - a;
    boolean improved = higherIsBetter ? delta > 0 : delta < 0;
    String marker = (delta == 0.0) ? " " : (improved ? "✓" : "✗");
    String beforeStr = String.format(fmt, a);
    String afterStr = String.format(fmt, b);
    String deltaStr = (delta >= 0 ? "+" : "") + String.format(fmt, delta);
    System.out.printf(
        "│ %-19s │ %13s │ %13s │ %s %11s │%n", label, beforeStr, afterStr, marker, deltaStr);
  }
}
