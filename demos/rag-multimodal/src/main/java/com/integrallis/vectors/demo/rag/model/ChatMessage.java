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

import java.time.Instant;
import java.util.List;

/**
 * Represents a chat message with cost tracking, timing, and source references.
 *
 * @param id Unique message identifier
 * @param role Message role (USER or ASSISTANT)
 * @param content Message text content
 * @param timestamp When the message was created
 * @param tokenCount Number of tokens in the message
 * @param costUsd Cost in USD for this message (for AI responses)
 * @param model LLM model used (for AI responses)
 * @param fromCache Whether the response came from cache
 * @param references Source references from retrieved content
 * @param elapsedMs Wall-clock duration of the RAG interaction in milliseconds (0 if not measured)
 * @param imageBytes Optional image data to display inline (null if no image)
 */
public record ChatMessage(
    String id,
    Role role,
    String content,
    Instant timestamp,
    int tokenCount,
    double costUsd,
    String model,
    boolean fromCache,
    List<Reference> references,
    long elapsedMs,
    byte[] imageBytes) {

  public enum Role {
    USER,
    ASSISTANT
  }

  /**
   * Creates a user message.
   *
   * @param content Message content
   * @return User chat message
   */
  public static ChatMessage user(String content) {
    return new ChatMessage(
        generateId(), Role.USER, content, Instant.now(), 0, 0.0, null, false, List.of(), 0, null);
  }

  /**
   * Creates an assistant message.
   *
   * @param content Message content
   * @param tokenCount Token count
   * @param costUsd Cost in USD
   * @param model Model name
   * @param fromCache Whether from cache
   * @return Assistant chat message
   */
  public static ChatMessage assistant(
      String content, int tokenCount, double costUsd, String model, boolean fromCache) {
    return new ChatMessage(
        generateId(),
        Role.ASSISTANT,
        content,
        Instant.now(),
        tokenCount,
        costUsd,
        model,
        fromCache,
        List.of(),
        0,
        null);
  }

  /**
   * Creates an assistant message with references.
   *
   * @param content Message content
   * @param tokenCount Token count
   * @param costUsd Cost in USD
   * @param model Model name
   * @param fromCache Whether from cache
   * @param references Source references
   * @return Assistant chat message with references
   */
  public static ChatMessage assistant(
      String content,
      int tokenCount,
      double costUsd,
      String model,
      boolean fromCache,
      List<Reference> references) {
    return new ChatMessage(
        generateId(),
        Role.ASSISTANT,
        content,
        Instant.now(),
        tokenCount,
        costUsd,
        model,
        fromCache,
        references,
        0,
        null);
  }

  /**
   * Returns a copy of this message with the elapsed time set.
   *
   * @param elapsed Duration in milliseconds
   * @return New ChatMessage with elapsed time
   */
  public ChatMessage withElapsedMs(long elapsed) {
    return new ChatMessage(
        id,
        role,
        content,
        timestamp,
        tokenCount,
        costUsd,
        model,
        fromCache,
        references,
        elapsed,
        imageBytes);
  }

  /**
   * Creates an assistant message that displays an inline image.
   *
   * @param caption Text caption for the image
   * @param imageData Raw image bytes to display
   * @param elapsedMs Duration in milliseconds
   * @return Assistant chat message with inline image
   */
  public static ChatMessage imageMessage(String caption, byte[] imageData, long elapsedMs) {
    return new ChatMessage(
        generateId(),
        Role.ASSISTANT,
        caption,
        Instant.now(),
        0,
        0.0,
        null,
        false,
        List.of(),
        elapsedMs,
        imageData);
  }

  /**
   * Creates a "thinking" indicator message shown while processing is in progress. Rendered with a
   * spinner by MessageBubble.
   *
   * @param content Description of what is being processed
   * @return Thinking chat message
   */
  public static ChatMessage thinking(String content) {
    return new ChatMessage(
        generateId(),
        Role.ASSISTANT,
        content,
        Instant.now(),
        0,
        0.0,
        "__thinking__",
        false,
        List.of(),
        0,
        null);
  }

  /** Returns true if this is a thinking/processing indicator message. */
  public boolean isThinking() {
    return "__thinking__".equals(model);
  }

  private static String generateId() {
    return java.util.UUID.randomUUID().toString();
  }
}
