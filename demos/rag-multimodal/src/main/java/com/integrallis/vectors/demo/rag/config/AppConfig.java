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
package com.integrallis.vectors.demo.rag.config;

import com.integrallis.vectors.demo.rag.model.LLMConfig;
import io.github.cdimascio.dotenv.Dotenv;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Application configuration loaded from application.properties.
 *
 * <p>Provides centralized access to configuration settings including API keys, vectors-server
 * connection, and LLM settings.
 */
public class AppConfig {

  private static final Pattern ENV_PLACEHOLDER =
      Pattern.compile("\\$\\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?}");

  private final Properties properties;
  private final Dotenv dotenv;

  private static AppConfig instance;

  private AppConfig() {
    properties = new Properties();
    String dotenvDir = System.getProperty("dotenv.directory", ".");
    dotenv =
        Dotenv.configure()
            .directory(dotenvDir)
            .ignoreIfMissing() // don't throw if .env absent
            .systemProperties() // also expose as system properties
            .load();
    loadProperties();
  }

  /**
   * Gets the singleton instance of AppConfig.
   *
   * @return AppConfig instance
   */
  public static synchronized AppConfig getInstance() {
    if (instance == null) {
      instance = new AppConfig();
    }
    return instance;
  }

  private void loadProperties() {
    try (InputStream input =
        getClass().getClassLoader().getResourceAsStream("application.properties")) {
      if (input == null) {
        System.err.println("Warning: application.properties not found. Using default values.");
        return;
      }
      properties.load(input);
    } catch (IOException e) {
      System.err.println("Error loading application.properties: " + e.getMessage());
    }
  }

  /**
   * Gets a property value, resolving {@code ${ENV_VAR:default}} placeholders against the .env file
   * and system environment variables (via dotenv-java).
   *
   * @param key Property key
   * @param defaultValue Default value if key not found
   * @return Property value with placeholders resolved
   */
  public String getProperty(String key, String defaultValue) {
    String raw = properties.getProperty(key, defaultValue);
    return resolvePlaceholders(raw);
  }

  /**
   * Resolves {@code ${VAR:default}} placeholders. Lookup order: .env file, then system environment,
   * then the placeholder default.
   */
  private String resolvePlaceholders(String value) {
    if (value == null || !value.contains("${")) {
      return value;
    }
    Matcher m = ENV_PLACEHOLDER.matcher(value);
    StringBuilder sb = new StringBuilder();
    while (m.find()) {
      String envKey = m.group(1);
      String fallback = m.group(2); // may be null
      // dotenv.get() checks .env first, then System.getenv()
      String resolved = dotenv.get(envKey);
      if (resolved == null) {
        resolved = (fallback != null) ? fallback : "";
      }
      m.appendReplacement(sb, Matcher.quoteReplacement(resolved));
    }
    m.appendTail(sb);
    return sb.toString();
  }

  /**
   * Gets the vectors-server base URL.
   *
   * @return vectors-server URL (e.g. http://localhost:8287)
   */
  public String getVectorsServerUrl() {
    return getProperty("vectors.server.url", "http://localhost:8287");
  }

  /**
   * Gets configured LLM provider.
   *
   * @return LLM provider
   */
  public LLMConfig.Provider getLLMProvider() {
    String provider = getProperty("llm.provider", "OPENAI");
    try {
      return LLMConfig.Provider.valueOf(provider.toUpperCase());
    } catch (IllegalArgumentException e) {
      System.err.println("Invalid provider: " + provider + ", using OPENAI");
      return LLMConfig.Provider.OPENAI;
    }
  }

  /**
   * Gets LLM configuration for the configured provider.
   *
   * @return LLMConfig
   */
  public LLMConfig getLLMConfig() {
    LLMConfig.Provider provider = getLLMProvider();
    String apiKey = getApiKeyForProvider(provider);
    String baseUrl = getBaseUrlForProvider(provider);
    String model = getModelForProvider(provider);
    double temperature =
        Double.parseDouble(getProperty(getProviderPrefix(provider) + ".temperature", "0.7"));
    int maxTokens =
        Integer.parseInt(getProperty(getProviderPrefix(provider) + ".max.tokens", "2048"));

    return new LLMConfig(provider, model, apiKey, baseUrl, maxTokens, temperature);
  }

  private String getApiKeyForProvider(LLMConfig.Provider provider) {
    String prefix = getProviderPrefix(provider);
    return getProperty(prefix + ".api.key", "");
  }

  private String getBaseUrlForProvider(LLMConfig.Provider provider) {
    String prefix = getProviderPrefix(provider);
    return switch (provider) {
      case OPENAI -> getProperty(prefix + ".base.url", "https://api.openai.com/v1");
      case ANTHROPIC -> getProperty(prefix + ".base.url", "https://api.anthropic.com");
      case AZURE -> getProperty("azure.endpoint", "");
      case OLLAMA -> getProperty(prefix + ".base.url", "http://localhost:11434");
    };
  }

  private String getModelForProvider(LLMConfig.Provider provider) {
    String prefix = getProviderPrefix(provider);
    return getProperty(prefix + ".model", provider.getDefaultModel());
  }

  private String getProviderPrefix(LLMConfig.Provider provider) {
    return provider.name().toLowerCase();
  }

  /**
   * Gets RAG max results.
   *
   * @return Max results
   */
  public int getRagMaxResults() {
    return Integer.parseInt(getProperty("rag.max.results", "5"));
  }

  /**
   * Gets RAG minimum score.
   *
   * @return Min score
   */
  public double getRagMinScore() {
    return Double.parseDouble(getProperty("rag.min.score", "0.7"));
  }

  /**
   * Gets PDF max pages to process.
   *
   * @return Max pages
   */
  public int getPdfMaxPages() {
    return Integer.parseInt(getProperty("pdf.max.pages", "500"));
  }

  /**
   * Gets the optional path to a local ONNX model file. If set, the model is loaded from this path
   * instead of downloading from HuggingFace.
   *
   * @return path string, or null if not configured
   */
  public String getOnnxModelPath() {
    String path = getProperty("onnx.model.path", null);
    return (path != null && !path.isBlank()) ? path : null;
  }

  /**
   * Validates that required configuration is present.
   *
   * @return true if valid
   */
  public boolean validateConfig() {
    LLMConfig.Provider provider = getLLMProvider();
    if (provider.requiresApiKey()) {
      String apiKey = getApiKeyForProvider(provider);
      if (apiKey == null || apiKey.isEmpty() || apiKey.contains("YOUR_")) {
        System.err.println(
            "Error: API key not configured for "
                + provider
                + ". Please update application.properties");
        return false;
      }
    }
    return true;
  }
}
