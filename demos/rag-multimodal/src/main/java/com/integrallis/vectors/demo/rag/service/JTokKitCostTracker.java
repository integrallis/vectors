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
import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import java.util.Map;

/**
 * Cost tracker implementation using JTokkit for token counting.
 *
 * <p>Provides accurate token counting for OpenAI models and cost calculation for various LLM
 * providers.
 */
public class JTokKitCostTracker implements CostTracker {

  private final EncodingRegistry registry;
  private final Encoding encoding;

  // Pricing per 1M tokens (as of October 2024)
  // These are INPUT token prices; output prices are typically higher
  private static final Map<String, Double> PRICING_PER_1M_TOKENS =
      Map.ofEntries(
          // OpenAI
          Map.entry("gpt-4o", 2.50),
          Map.entry("gpt-4o-mini", 0.15),
          Map.entry("gpt-4-turbo", 10.00),
          Map.entry("gpt-4", 30.00),
          Map.entry("gpt-3.5-turbo", 0.50),
          // Anthropic
          Map.entry("claude-3-5-sonnet-20241022", 3.00),
          Map.entry("claude-3-5-haiku-20241022", 0.80),
          Map.entry("claude-3-opus-20240229", 15.00),
          Map.entry("claude-3-sonnet-20240229", 3.00),
          Map.entry("claude-3-haiku-20240307", 0.25),
          // Azure uses same pricing as OpenAI
          Map.entry("azure-gpt-4o", 2.50),
          Map.entry("azure-gpt-4-turbo", 10.00));

  private static final double DEFAULT_PRICE_PER_1M = 1.00;

  public JTokKitCostTracker() {
    this.registry = Encodings.newDefaultEncodingRegistry();
    // Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo, etc.)
    this.encoding = registry.getEncoding(EncodingType.CL100K_BASE);
  }

  @Override
  public int countTokens(String text) {
    if (text == null || text.isEmpty()) {
      return 0;
    }
    return encoding.countTokens(text);
  }

  @Override
  public double calculateCost(LLMConfig.Provider provider, String model, int tokens) {
    if (tokens <= 0) {
      return 0.0;
    }

    // Ollama is free (local)
    if (provider == LLMConfig.Provider.OLLAMA) {
      return 0.0;
    }

    double costPer1M = getCostPer1KTokens(provider, model) * 1000;
    return (tokens / 1_000_000.0) * costPer1M;
  }

  @Override
  public double getCostPer1KTokens(LLMConfig.Provider provider, String model) {
    // Ollama is free
    if (provider == LLMConfig.Provider.OLLAMA) {
      return 0.0;
    }

    // For Azure, normalize model name
    String pricingKey = model;
    if (provider == LLMConfig.Provider.AZURE && !model.startsWith("azure-")) {
      pricingKey = "azure-" + model;
    }

    // Get pricing, use default if not found
    double costPer1M = PRICING_PER_1M_TOKENS.getOrDefault(pricingKey, DEFAULT_PRICE_PER_1M);
    return costPer1M / 1000.0; // Convert to per 1K
  }

  /**
   * Formats cost as a human-readable string.
   *
   * @param costUsd Cost in USD
   * @return Formatted cost string
   */
  public static String formatCost(double costUsd) {
    if (costUsd == 0.0) {
      return "$0.00";
    } else if (costUsd < 0.01) {
      return String.format("$%.4f", costUsd);
    } else {
      return String.format("$%.2f", costUsd);
    }
  }

  /**
   * Formats token count as a human-readable string.
   *
   * @param tokens Token count
   * @return Formatted token string
   */
  public static String formatTokens(int tokens) {
    if (tokens < 1000) {
      return tokens + " tokens";
    } else if (tokens < 1_000_000) {
      return String.format("%.1fK tokens", tokens / 1000.0);
    } else {
      return String.format("%.2fM tokens", tokens / 1_000_000.0);
    }
  }
}
