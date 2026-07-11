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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ProviderRegistryTest {

  private static final EmbeddingProviderSpec OPENAI =
      new EmbeddingProviderSpec(
          "openai",
          "openai",
          "https://api.openai.com/v1",
          "OPENAI_API_KEY",
          List.of("text-embedding-3-small", "text-embedding-3-large"));
  private static final EmbeddingProviderSpec OLLAMA =
      new EmbeddingProviderSpec(
          "ollama",
          "openai-compatible",
          "http://localhost:11434/v1",
          null,
          List.of("mxbai-embed-large"));

  private static ProviderRegistry registry(Function<String, String> env) {
    return new ProviderRegistry(List.of(OPENAI, OLLAMA), env);
  }

  @Test
  void bundledCatalogLoadsTwoProviders() {
    // load() reads the bundled embedding-providers.json off the classpath (no override, no
    // network).
    ProviderRegistry reg = ProviderRegistry.load(null);
    assertThat(reg.statuses()).extracting(ProviderStatus::id).containsExactly("openai", "ollama");
  }

  @Test
  void statusReportsKeyPresentWhenEnvSet() {
    ProviderRegistry reg = registry(Map.of("OPENAI_API_KEY", "sk-test")::get);
    Map<String, ProviderStatus> byId = byId(reg);
    // openai: key env resolves non-blank → present. ollama: no key required → always present.
    assertThat(byId.get("openai").keyPresent()).isTrue();
    assertThat(byId.get("openai").apiKeyEnv()).isEqualTo("OPENAI_API_KEY");
    assertThat(byId.get("ollama").keyPresent()).isTrue();
    assertThat(byId.get("ollama").apiKeyEnv()).isNull();
  }

  @Test
  void statusReportsKeyMissingWhenEnvUnset() {
    ProviderRegistry reg = registry(name -> null); // nothing resolves
    Map<String, ProviderStatus> byId = byId(reg);
    assertThat(byId.get("openai").keyPresent()).isFalse();
    // A key-less provider is still ready even when the resolver returns nothing.
    assertThat(byId.get("ollama").keyPresent()).isTrue();
  }

  @Test
  void statusNeverExposesKeyValue() {
    ProviderRegistry reg = registry(Map.of("OPENAI_API_KEY", "sk-secret-value")::get);
    assertThat(reg.statuses().toString()).doesNotContain("sk-secret-value");
  }

  @Test
  void providerForMatchesByModelAndKeyPresence() {
    ProviderRegistry withKey = registry(Map.of("OPENAI_API_KEY", "sk-test")::get);
    // Matches an openai model when the key is present.
    assertThat(withKey.providerFor("text-embedding-3-small", null)).isPresent();
    // Matches a key-less provider's model regardless of env.
    assertThat(withKey.providerFor("mxbai-embed-large", null)).isPresent();
    // Unknown model → no provider.
    assertThat(withKey.providerFor("no-such-model", null)).isEmpty();
  }

  @Test
  void providerForSkipsProviderWithMissingKey() {
    ProviderRegistry noKey = registry(name -> null);
    // openai model requires a key that is absent → no provider.
    assertThat(noKey.providerFor("text-embedding-3-small", null)).isEmpty();
    // ollama needs no key → still resolvable.
    assertThat(noKey.providerFor("mxbai-embed-large", null)).isPresent();
  }

  private static Map<String, ProviderStatus> byId(ProviderRegistry reg) {
    return reg.statuses().stream()
        .collect(java.util.stream.Collectors.toMap(ProviderStatus::id, s -> s));
  }
}
