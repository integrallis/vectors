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

/**
 * Hyperparameter optimization for the {@code vectors-*} retrieval stack.
 *
 * <p>The module ships three studies — an index study (HNSW / Vamana / IVF index parameters,
 * search-time knobs, quantizer choices, similarity metric), a router-threshold study, and a
 * cache-threshold study — driven by a Java-native sampler suite (Grid, Random, univariate TPE).
 * Trials produce a {@link com.integrallis.vectors.optimizer.study.TrialResult} with retrieval
 * metrics ({@link com.integrallis.vectors.optimizer.eval.Metrics}) plus latency / build-time /
 * memory and a composite scalar from {@link com.integrallis.vectors.optimizer.objective.Objective}.
 *
 * <p>Inspired by {@code redis-retrieval-optimizer} (Python, Optuna-backed) and the threshold
 * optimizer helper in {@code redis-vl-python}; this module re-implements the sampling stack in
 * Java with no external HPO dependency.
 */
package com.integrallis.vectors.optimizer;
