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

import com.integrallis.vectors.demo.rag.config.AppConfig;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Registry of available LLM providers and models.
 *
 * <p>Filters providers based on API key availability in environment variables or
 * application.properties.
 */
public class ModelRegistry {

  private static final Map<LLMConfig.Provider, List<String>> PROVIDER_MODELS =
      new EnumMap<>(LLMConfig.Provider.class);

  static {
    // OpenAI models
    PROVIDER_MODELS.put(
        LLMConfig.Provider.OPENAI,
        List.of("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"));

    // Anthropic models
    PROVIDER_MODELS.put(
        LLMConfig.Provider.ANTHROPIC,
        List.of(
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"));

    // Azure OpenAI
    PROVIDER_MODELS.put(LLMConfig.Provider.AZURE, List.of("gpt-4o", "gpt-4", "gpt-35-turbo"));

    // Ollama - local models
    PROVIDER_MODELS.put(
        LLMConfig.Provider.OLLAMA,
        List.of("llama3.2-vision", "llama3.2", "llama3.1", "mistral", "mixtral"));
  }

  /**
   * Gets providers that have valid API keys configured.
   *
   * @return List of available providers
   */
  public static List<LLMConfig.Provider> getAvailableProviders() {
    List<LLMConfig.Provider> available = new ArrayList<>();

    for (LLMConfig.Provider provider : LLMConfig.Provider.values()) {
      if (isProviderAvailable(provider)) {
        available.add(provider);
      }
    }

    return available;
  }

  /**
   * Checks if a provider is available based on API key.
   *
   * @param provider Provider to check
   * @return true if available
   */
  public static boolean isProviderAvailable(LLMConfig.Provider provider) {
    if (!provider.requiresApiKey()) {
      return true; // Ollama doesn't require API key
    }

    String apiKey = getApiKeyForProvider(provider);
    return apiKey != null && !apiKey.isEmpty() && !apiKey.startsWith("YOUR_");
  }

  /**
   * Gets API key for a provider from environment variables or application.properties.
   *
   * @param provider Provider
   * @return API key or null
   */
  public static String getApiKeyForProvider(LLMConfig.Provider provider) {
    return switch (provider) {
      case OPENAI -> getApiKey("OPENAI_API_KEY", "openai.api.key");
      case ANTHROPIC -> getApiKey("ANTHROPIC_API_KEY", "anthropic.api.key");
      case AZURE -> getApiKey("AZURE_OPENAI_API_KEY", "azure.api.key");
      case OLLAMA -> null; // No API key needed
    };
  }

  private static String getApiKey(String envVar, String property) {
    // AppConfig.getProperty resolves ${ENV_VAR:default} placeholders against
    // dotenv (.env file) and System.getenv(), so it covers all sources.
    AppConfig config = AppConfig.getInstance();
    String value = config.getProperty(property, null);
    if (value != null && !value.isEmpty()) {
      return value;
    }
    // Direct env var check as fallback (e.g. if property not in application.properties)
    value = System.getenv(envVar);
    if (value != null && !value.isEmpty()) {
      return value;
    }
    return null;
  }

  /**
   * Gets models available for a provider.
   *
   * @param provider Provider
   * @return List of model names
   */
  public static List<String> getModelsForProvider(LLMConfig.Provider provider) {
    return PROVIDER_MODELS.getOrDefault(provider, List.of(provider.getDefaultModel()));
  }

  /**
   * Gets the default model for a provider.
   *
   * @param provider Provider
   * @return Default model name
   */
  public static String getDefaultModel(LLMConfig.Provider provider) {
    List<String> models = getModelsForProvider(provider);
    return models.isEmpty() ? provider.getDefaultModel() : models.get(0);
  }
}
