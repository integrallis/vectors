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
package com.integrallis.vectors.demo.rag.service;

import com.integrallis.vectors.demo.rag.model.LLMConfig;

/** Interface for tracking token counts and costs. */
public interface CostTracker {

  /**
   * Counts tokens in text.
   *
   * @param text Text to count tokens
   * @return Token count
   */
  int countTokens(String text);

  /**
   * Calculates cost for a given number of tokens.
   *
   * @param provider LLM provider
   * @param model Model name
   * @param tokens Number of tokens
   * @return Cost in USD
   */
  double calculateCost(LLMConfig.Provider provider, String model, int tokens);

  /**
   * Gets the cost per 1K tokens for a model.
   *
   * @param provider LLM provider
   * @param model Model name
   * @return Cost per 1K tokens in USD
   */
  double getCostPer1KTokens(LLMConfig.Provider provider, String model);
}
