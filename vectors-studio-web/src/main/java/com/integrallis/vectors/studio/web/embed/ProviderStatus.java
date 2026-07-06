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

import java.util.List;

/**
 * Redacted, UI-safe view of a configured embedding provider. Reports whether an API key is present
 * ({@code keyPresent}) without ever exposing the key value; {@code apiKeyEnv} is the name of the
 * environment variable that supplies it (or {@code null} when the endpoint needs no key).
 *
 * @param id provider id
 * @param type {@code openai} or {@code openai-compatible}
 * @param models embedding model ids this provider can serve
 * @param keyPresent {@code true} when no key is required or the configured env var resolves
 *     non-blank
 * @param apiKeyEnv name of the env var holding the key, or {@code null} when none is required
 */
public record ProviderStatus(
    String id, String type, List<String> models, boolean keyPresent, String apiKeyEnv) {

  public ProviderStatus {
    models = models == null ? List.of() : List.copyOf(models);
  }
}
