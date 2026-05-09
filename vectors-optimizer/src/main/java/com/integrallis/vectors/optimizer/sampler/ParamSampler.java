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
import java.util.List;

/**
 * Strategy for proposing the next {@link Trial}, given the history of completed trials and their
 * (maximize-direction) objective scores.
 */
public sealed interface ParamSampler permits GridSampler, RandomSampler, TpeSampler {

  /**
   * Proposes the next trial.
   *
   * @param history previously completed trials, in submission order; may be empty for the first
   *     call
   * @return the next trial to evaluate
   * @throws NoMoreTrialsException if the sampler has exhausted its space (only thrown by
   *     {@link GridSampler})
   */
  Trial next(List<? extends ScoredTrial> history);
}
