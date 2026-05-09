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

/**
 * Default entry point — prints the tutorial map. Each stage has its own Gradle task; running {@code
 * :demos:optimizer-tutorial:run} invokes this class.
 */
public final class TutorialApp {

  private TutorialApp() {}

  public static void main(String[] args) {
    System.out.println("vectors-optimizer tutorial — four stages, one dataset, before vs. after.");
    System.out.println();
    System.out.println("  ./gradlew :demos:optimizer-tutorial:stage1Baseline");
    System.out.println("    Build with default HNSW (M=16, efConstruction=200); record metrics.");
    System.out.println();
    System.out.println("  ./gradlew :demos:optimizer-tutorial:stage2BroadSweep");
    System.out.println("    20-trial Random search across m & efConstruction.");
    System.out.println();
    System.out.println("  ./gradlew :demos:optimizer-tutorial:stage3Tpe");
    System.out.println("    40-trial TPE study; side-by-side comparison vs. baseline.");
    System.out.println();
    System.out.println("  ./gradlew :demos:optimizer-tutorial:stage4Threshold");
    System.out.println("    Sweep router decision thresholds on a labelled probe set.");
    System.out.println();
    System.out.println(
        "Dataset:    set VECTORS_OPTIMIZER_TUTORIAL_DATASET=<name> to override the default");
    System.out.println(
        "            (default: " + TutorialDataset.DEFAULT_DATASET + ", auto-downloaded).");
    System.out.println(
        "Subsample:  set VECTORS_OPTIMIZER_TUTORIAL_FULL=1 to run on the full dataset");
    System.out.println(
        "            (default subsample: "
            + TutorialDataset.DEFAULT_CORPUS_LIMIT
            + " corpus / "
            + TutorialDataset.DEFAULT_QUERY_LIMIT
            + " queries).");
    System.out.println("Output:     ~/.vectors/optimizer/tutorial/{baseline.json,best.json}");
    System.out.println("            ~/.vectors/optimizer/studies/<study-id>.jsonl");
  }
}
