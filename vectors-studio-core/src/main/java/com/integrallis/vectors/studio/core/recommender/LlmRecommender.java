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
package com.integrallis.vectors.studio.core.recommender;

import dev.langchain4j.model.chat.ChatModel;
import java.util.Objects;

/**
 * Optional LLM-backed enricher that adds a human-readable narrative to a {@link
 * ProjectionRecommendation}. Constructed only when langchain4j is present on the classpath.
 */
public final class LlmRecommender {

  private final ChatModel model;

  public LlmRecommender(ChatModel model) {
    this.model = Objects.requireNonNull(model, "model");
  }

  /** Returns a copy of {@code base} with {@code llmExplanation} populated. */
  public ProjectionRecommendation enrich(DatasetStats stats, ProjectionRecommendation base) {
    String prompt =
        """
        Dataset stats: N=%d, dim=%d, intrinsicDim≈%.1f, labels=%s, hasText=%s, sparsity=%.2f.
        Heuristic chose: %s with params %s because %s.
        Briefly explain to a data analyst why this is a reasonable starting point and suggest
        ONE parameter to consider tweaking.
        """
            .formatted(
                stats.n(),
                stats.dim(),
                stats.intrinsicDimEstimate(),
                stats.labelCardinality(),
                stats.hasText(),
                stats.sparsityRatio(),
                base.algorithm(),
                base.params(),
                base.rationale());
    String reply = model.chat(prompt);
    return new ProjectionRecommendation(base.algorithm(), base.params(), base.rationale(), reply);
  }
}
