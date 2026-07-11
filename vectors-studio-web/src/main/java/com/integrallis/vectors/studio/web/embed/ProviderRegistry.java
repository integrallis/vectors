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
package com.integrallis.vectors.studio.web.embed;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.vectors.optimizer.embed.EmbeddingProvider;
import com.integrallis.vectors.studio.web.EnvDefaults;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads the bundled {@code embedding-providers.json} catalog from the classpath and merges an
 * optional {@code VECTORS_STUDIO_PROVIDERS} override file by {@code id} (mirrors {@code
 * DatasetCatalog}). Resolves query-time {@link EmbeddingProvider}s by model and reports redacted
 * per-provider {@link ProviderStatus}. API key values are resolved lazily via an env resolver and
 * are never persisted or exposed.
 */
public class ProviderRegistry {

  /** Env var pointing at a user-supplied provider catalog to merge over the bundled defaults. */
  public static final String OVERRIDE_ENV = "VECTORS_STUDIO_PROVIDERS";

  private static final Logger LOG = LoggerFactory.getLogger(ProviderRegistry.class);
  private static final String BUNDLED_RESOURCE = "/embedding-providers.json";
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final List<EmbeddingProviderSpec> specs;
  private final Function<String, String> envResolver;

  /**
   * @param specs provider specs in declaration order
   * @param envResolver resolves an env-var name to its value (non-blank when present), or {@code
   *     null}; used both for key-presence checks and to bind resolved keys into providers
   */
  public ProviderRegistry(List<EmbeddingProviderSpec> specs, Function<String, String> envResolver) {
    this.specs = List.copyOf(specs);
    this.envResolver = envResolver;
  }

  /**
   * Loads the bundled catalog and merges the optional {@code VECTORS_STUDIO_PROVIDERS} override.
   */
  public static ProviderRegistry load() {
    return load(System.getenv(OVERRIDE_ENV));
  }

  /**
   * Loads the bundled catalog and merges the override file at {@code overridePath} (may be null).
   */
  public static ProviderRegistry load(String overridePath) {
    Map<String, EmbeddingProviderSpec> merged = new LinkedHashMap<>();
    for (EmbeddingProviderSpec s : readBundled()) {
      merged.put(s.id(), s);
    }
    if (overridePath != null && !overridePath.isBlank()) {
      for (EmbeddingProviderSpec s : readOverride(Path.of(overridePath.trim()))) {
        merged.put(s.id(), s);
      }
    }
    EnvDefaults env = EnvDefaults.fromRepoRoot();
    return new ProviderRegistry(new ArrayList<>(merged.values()), env::get);
  }

  private static List<EmbeddingProviderSpec> readBundled() {
    try (InputStream in = ProviderRegistry.class.getResourceAsStream(BUNDLED_RESOURCE)) {
      if (in == null) {
        LOG.warn("studio: bundled provider catalog {} not found on classpath", BUNDLED_RESOURCE);
        return List.of();
      }
      return parse(MAPPER.readTree(in));
    } catch (IOException e) {
      LOG.warn("studio: failed to read bundled provider catalog: {}", e.getMessage());
      return List.of();
    }
  }

  private static List<EmbeddingProviderSpec> readOverride(Path path) {
    if (!Files.isReadable(path)) {
      LOG.warn("studio: {} points at unreadable file {}", OVERRIDE_ENV, path);
      return List.of();
    }
    try (InputStream in = Files.newInputStream(path)) {
      List<EmbeddingProviderSpec> extra = parse(MAPPER.readTree(in));
      LOG.info("studio: merged {} provider entries from {}", extra.size(), path);
      return extra;
    } catch (IOException e) {
      LOG.warn("studio: failed to read {} file {}: {}", OVERRIDE_ENV, path, e.getMessage());
      return List.of();
    }
  }

  private static List<EmbeddingProviderSpec> parse(JsonNode root) {
    JsonNode providers = root == null ? null : root.get("providers");
    if (providers == null || !providers.isArray()) {
      return List.of();
    }
    List<EmbeddingProviderSpec> out = new ArrayList<>(providers.size());
    for (JsonNode n : providers) {
      String id = text(n, "id", null);
      if (id == null || id.isBlank()) {
        continue; // an entry without an id cannot be a merge key; skip defensively
      }
      List<String> models = new ArrayList<>();
      JsonNode arr = n.get("models");
      if (arr != null && arr.isArray()) {
        for (JsonNode m : arr) {
          models.add(m.asText());
        }
      }
      out.add(
          new EmbeddingProviderSpec(
              id,
              text(n, "type", "openai-compatible"),
              text(n, "baseUrl", null),
              text(n, "apiKeyEnv", null),
              models));
    }
    return out;
  }

  private static String text(JsonNode n, String field, String dflt) {
    JsonNode v = n.get(field);
    return v == null || v.isNull() ? dflt : v.asText();
  }

  /** Redacted status for every configured provider (never exposes key values). */
  public List<ProviderStatus> statuses() {
    List<ProviderStatus> out = new ArrayList<>(specs.size());
    for (EmbeddingProviderSpec s : specs) {
      out.add(new ProviderStatus(s.id(), s.type(), s.models(), keyPresent(s), s.apiKeyEnv()));
    }
    return out;
  }

  /**
   * First provider whose {@code models} contains {@code model} and whose API key is present, bound
   * to an {@link HttpEmbeddingProvider} at (baseUrl, resolved key, model, dimensions). Empty when
   * no such provider is configured.
   */
  public Optional<EmbeddingProvider> providerFor(String model, Integer dimensions) {
    if (model == null) {
      return Optional.empty();
    }
    for (EmbeddingProviderSpec s : specs) {
      if (s.models().contains(model) && keyPresent(s)) {
        String apiKey = s.apiKeyEnv() == null ? null : resolve(s.apiKeyEnv());
        return Optional.of(
            new HttpEmbeddingProvider(s.baseUrl(), apiKey, model, dimensions, MAPPER));
      }
    }
    return Optional.empty();
  }

  private boolean keyPresent(EmbeddingProviderSpec spec) {
    if (spec.apiKeyEnv() == null) {
      return true;
    }
    String v = resolve(spec.apiKeyEnv());
    return v != null && !v.isBlank();
  }

  private String resolve(String envName) {
    return envResolver == null ? null : envResolver.apply(envName);
  }
}
