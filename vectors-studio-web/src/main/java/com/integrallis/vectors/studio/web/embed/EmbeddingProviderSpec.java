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
 * One entry in the embedding-provider catalog (a row of {@code embedding-providers.json}).
 * Describes an OpenAI-compatible embeddings endpoint and the models it can serve. The API key value
 * is never stored here; only the name of the environment variable ({@code apiKeyEnv}) that holds it
 * — which may be {@code null} when the endpoint requires no authentication (e.g. a local Ollama
 * server).
 *
 * @param id stable provider identifier (also the merge key for user overrides)
 * @param type {@code openai} or {@code openai-compatible}; informational only
 * @param baseUrl API base URL, e.g. {@code https://api.openai.com/v1}
 * @param apiKeyEnv environment-variable name holding the API key, or {@code null} if none required
 * @param models embedding model ids this provider can serve
 */
public record EmbeddingProviderSpec(
    String id, String type, String baseUrl, String apiKeyEnv, List<String> models) {

  public EmbeddingProviderSpec {
    models = models == null ? List.of() : List.copyOf(models);
  }
}
