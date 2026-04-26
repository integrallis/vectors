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

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a field to be automatically wrapped with VCR recording/playback functionality.
 *
 * <p>When applied to an {@code EmbeddingModel} or {@code ChatModel} field, the VCR extension/
 * listener will automatically wrap the model with the appropriate VCR wrapper after it has been
 * initialized by the test's {@code @BeforeEach} / {@code @BeforeMethod} method.
 *
 * <p>Supported model types:
 *
 * <ul>
 *   <li>LangChain4j: {@code dev.langchain4j.model.embedding.EmbeddingModel}, {@code
 *       dev.langchain4j.model.chat.ChatModel}
 *   <li>Spring AI: {@code org.springframework.ai.embedding.EmbeddingModel}, {@code
 *       org.springframework.ai.chat.model.ChatModel}
 * </ul>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface VCRModel {

  /**
   * Optional model name for embedding cache key generation. If not specified, the field name will
   * be used.
   */
  String modelName() default "";
}
