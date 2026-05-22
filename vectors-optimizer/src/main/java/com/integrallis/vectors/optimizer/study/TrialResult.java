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

import com.integrallis.vectors.optimizer.space.ScoredTrial;
import com.integrallis.vectors.optimizer.space.Trial;
import java.time.Instant;

/**
 * Outcome of a single trial: parameters, timing window, IR metrics, latency percentiles, build
 * cost, memory footprint, and the composite objective score (higher is better) used by samplers to
 * direct the search.
 */
public record TrialResult(
    Trial trial,
    Instant startedAt,
    Instant finishedAt,
    double recallAtK,
    double ndcgAtK,
    double precisionAtK,
    double f1AtK,
    double mrr,
    double latencyP50Us,
    double latencyP95Us,
    double latencyP99Us,
    long buildTimeMs,
    long memoryBytes,
    double objectiveScore)
    implements ScoredTrial {}
