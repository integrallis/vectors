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
package com.integrallis.vectors.core;

import java.util.Objects;

/** Reusable ordered stages that share one configured GGUF worker publication. */
public final class GgufStagePlan {

  private final Stage[] stages;

  private GgufStagePlan(Stage[] stages) {
    this.stages = stages;
  }

  /** Creates an immutable plan containing at least one stage. */
  public static GgufStagePlan of(Stage... stages) {
    Objects.requireNonNull(stages, "stages");
    if (stages.length == 0) {
      throw new IllegalArgumentException("a GGUF stage plan requires at least one stage");
    }
    Stage[] copy = stages.clone();
    for (Stage stage : copy) {
      Objects.requireNonNull(stage, "stage");
    }
    return new GgufStagePlan(copy);
  }

  /** Creates one range-partitioned stage. */
  public static Stage stage(int workItems, RangeOperation operation) {
    return new Stage(workItems, operation);
  }

  /** Executes this plan using the configured GGUF execution policy. */
  public void execute() {
    GgufParallelSupport.execute(this);
  }

  /** Returns the number of ordered stages. */
  public int stageCount() {
    return stages.length;
  }

  Stage stage(int index) {
    return stages[index];
  }

  void executeSerially() {
    for (Stage stage : stages) {
      stage.operation().execute(0, stage.workItems());
    }
  }

  /** One ordered stage partitioned into non-overlapping contiguous ranges. */
  public record Stage(int workItems, RangeOperation operation) {

    public Stage {
      if (workItems < 1) {
        throw new IllegalArgumentException("stage workItems must be positive");
      }
      operation = Objects.requireNonNull(operation, "operation");
    }
  }

  /** Performs one non-empty half-open range of stage work items. */
  @FunctionalInterface
  public interface RangeOperation {

    void execute(int fromInclusive, int toExclusive);
  }
}
