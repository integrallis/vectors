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
package com.integrallis.vectors.demo.rag.model;

/**
 * Configuration for LLM provider.
 *
 * @param provider LLM provider type
 * @param model Model name
 * @param apiKey API key
 * @param baseUrl Base URL for the provider
 * @param maxTokens Maximum tokens for generation
 * @param temperature Temperature for generation
 */
public record LLMConfig(
    Provider provider,
    String model,
    String apiKey,
    String baseUrl,
    int maxTokens,
    double temperature) {

  public enum Provider {
    OPENAI("OpenAI", "gpt-4o", true),
    ANTHROPIC("Anthropic", "claude-3-5-sonnet-20241022", true),
    AZURE("Azure OpenAI", "gpt-4o", true),
    OLLAMA("Ollama", "llama3.2-vision", false);

    private final String displayName;
    private final String defaultModel;
    private final boolean requiresApiKey;

    Provider(String displayName, String defaultModel, boolean requiresApiKey) {
      this.displayName = displayName;
      this.defaultModel = defaultModel;
      this.requiresApiKey = requiresApiKey;
    }

    public String getDisplayName() {
      return displayName;
    }

    public String getDefaultModel() {
      return defaultModel;
    }

    public boolean requiresApiKey() {
      return requiresApiKey;
    }
  }

  /**
   * Creates default configuration for a provider.
   *
   * @param provider LLM provider
   * @param apiKey API key (can be null for local models)
   * @return Default configuration
   */
  public static LLMConfig defaultConfig(Provider provider, String apiKey) {
    String baseUrl =
        switch (provider) {
          case OPENAI -> "https://api.openai.com/v1";
          case ANTHROPIC -> "https://api.anthropic.com";
          case AZURE -> null; // Configured separately
          case OLLAMA -> "http://localhost:11434";
        };

    return new LLMConfig(provider, provider.getDefaultModel(), apiKey, baseUrl, 2048, 0.7);
  }
}
