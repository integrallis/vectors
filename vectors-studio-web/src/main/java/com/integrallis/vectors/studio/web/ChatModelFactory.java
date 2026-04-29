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
package com.integrallis.vectors.studio.web;

import com.integrallis.vectors.studio.core.recommender.LlmRecommender;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Optional builder for an {@link LlmRecommender}. Returns {@link Optional#empty()} unless a
 * langchain4j {@code ChatModel} is found on the classpath via reflection.
 *
 * <p>v1 ships with no built-in provider wiring: applications that want LLM enrichment must add
 * their own langchain4j provider (e.g. {@code dev.langchain4j:langchain4j-open-ai}) and pass the
 * constructed {@code ChatModel} to {@link LlmRecommender} directly.
 */
public final class ChatModelFactory {

  private static final Logger LOG = LoggerFactory.getLogger(ChatModelFactory.class);

  private ChatModelFactory() {}

  /**
   * Returns an empty {@link Optional} unless a non-null {@code ChatModel} is supplied by the
   * caller. Reserved for future SPI-driven discovery (system property, JNDI, etc.).
   */
  public static Optional<LlmRecommender> create(Object chatModelOrNull) {
    if (chatModelOrNull == null) return Optional.empty();
    try {
      Class<?> chatModelClass = Class.forName("dev.langchain4j.model.chat.ChatModel");
      if (!chatModelClass.isInstance(chatModelOrNull)) {
        LOG.warn(
            "ChatModelFactory: supplied object is not a langchain4j ChatModel ({})",
            chatModelOrNull.getClass().getName());
        return Optional.empty();
      }
      return Optional.of(
          new LlmRecommender(
              (dev.langchain4j.model.chat.ChatModel) chatModelClass.cast(chatModelOrNull)));
    } catch (ClassNotFoundException e) {
      LOG.info("langchain4j-core not on classpath; LLM enrichment disabled");
      return Optional.empty();
    }
  }
}
