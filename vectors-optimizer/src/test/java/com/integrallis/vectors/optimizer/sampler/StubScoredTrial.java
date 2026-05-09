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
package com.integrallis.vectors.optimizer.sampler;

import com.integrallis.vectors.optimizer.space.ScoredTrial;
import com.integrallis.vectors.optimizer.space.Trial;

/** Test-only ScoredTrial implementation used until TrialResult lands in Phase 3. */
record StubScoredTrial(Trial trial, double objectiveScore) implements ScoredTrial {}
