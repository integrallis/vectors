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
package com.integrallis.vectors.vcr;

/**
 * SPI for wrapping framework-specific LLM model instances with a VCR interceptor.
 *
 * <p>Implementations are discovered via {@link java.util.ServiceLoader}. Each adapter module (e.g.
 * {@code vectors-vcr-langchain4j}, {@code vectors-vcr-spring-ai}) registers one or more providers
 * that declare which model types they can handle.
 *
 * <p>A provider that cannot handle the given model returns {@code null}, allowing {@link
 * VCRModelWrapper} to try the next provider.
 */
public interface ModelWrapperProvider {

  /**
   * Attempts to wrap {@code model} with a VCR interceptor.
   *
   * @param model the model instance to wrap (never null)
   * @param testId the current test identifier
   * @param mode the effective VCR mode
   * @param modelName a human-readable model name used for cache keys
   * @param cassetteStore the cassette store for persistence
   * @return the wrapped model, or {@code null} if this provider does not support {@code model}
   */
  Object wrap(
      Object model, String testId, VCRMode mode, String modelName, CassetteStore cassetteStore);

  /**
   * @return a diagnostic name for this provider (used in logging)
   */
  default String name() {
    return getClass().getSimpleName();
  }
}
