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
package com.integrallis.vectors.optimizer.space;

/**
 * Read-only view of a finished trial: its parameters and a single composite score to be
 * maximized. {@code TrialResult} (Phase 3) is the canonical implementation; samplers consume
 * histories via this narrow interface so they do not depend on the heavier {@code study} package.
 */
public interface ScoredTrial {

  /** The trial's parameters as drawn from the search space. */
  Trial trial();

  /** Composite objective score; higher is better. */
  double objectiveScore();
}
