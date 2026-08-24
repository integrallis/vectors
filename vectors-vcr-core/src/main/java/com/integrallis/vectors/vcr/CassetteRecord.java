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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Sealed hierarchy of recorded cassette payloads.
 *
 * <p>Three payload shapes are supported:
 *
 * <ul>
 *   <li>{@link Embedding} — a single dense float vector plus metadata
 *   <li>{@link BatchEmbedding} — a batch of dense float vectors plus metadata
 *   <li>{@link Chat} — a chat prompt/response pair plus metadata
 * </ul>
 */
public sealed interface CassetteRecord
    permits CassetteRecord.Embedding, CassetteRecord.BatchEmbedding, CassetteRecord.Chat {

  /**
   * @return the test identifier that produced this record
   */
  String testId();

  /**
   * @return the model name (human-readable, used for metadata only)
   */
  String model();

  /**
   * @return the epoch-milli timestamp at which the record was produced
   */
  long timestamp();

  /**
   * @return the SHA-256 signature of the complete model request, or {@code null} for a legacy
   *     cassette
   */
  String requestSignature();

  /** A single embedding vector. */
  record Embedding(
      String testId, String model, long timestamp, float[] embedding, String requestSignature)
      implements CassetteRecord {
    /** Source-compatible constructor for legacy/manual cassettes without a request signature. */
    public Embedding(String testId, String model, long timestamp, float[] embedding) {
      this(testId, model, timestamp, embedding, null);
    }

    /** Compact constructor with null/shape checks. */
    public Embedding {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(embedding, "embedding");
    }
  }

  /** A batch of embedding vectors produced by a single batch call. */
  record BatchEmbedding(
      String testId, String model, long timestamp, float[][] embeddings, String requestSignature)
      implements CassetteRecord {
    /** Source-compatible constructor for legacy/manual cassettes without a request signature. */
    public BatchEmbedding(String testId, String model, long timestamp, float[][] embeddings) {
      this(testId, model, timestamp, embeddings, null);
    }

    /** Compact constructor with null/shape checks. */
    public BatchEmbedding {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(embeddings, "embeddings");
    }
  }

  /** A chat exchange with enough structured response data for lossless framework playback. */
  record Chat(
      String testId,
      String model,
      long timestamp,
      String prompt,
      ChatPayload response,
      String requestSignature)
      implements CassetteRecord {
    /** Source-compatible constructor for legacy/manual cassettes without a request signature. */
    public Chat(String testId, String model, long timestamp, String prompt, ChatPayload response) {
      this(testId, model, timestamp, prompt, response, null);
    }

    /** Compact constructor validates non-null prompt and response payloads. */
    public Chat {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(prompt, "prompt");
      Objects.requireNonNull(response, "response");
    }
  }

  /** Structured chat response payload. */
  record ChatPayload(
      AiMessagePayload aiMessage,
      ChatMetadata metadata,
      GenerationMetadata generationMetadata,
      List<ChatGenerationPayload> additionalGenerations,
      List<StreamEvent> streamEvents,
      List<ChatPayload> streamChunks) {
    /** Constructor for an ordinary single-generation blocking response. */
    public ChatPayload(AiMessagePayload aiMessage, ChatMetadata metadata) {
      this(aiMessage, metadata, GenerationMetadata.empty(), List.of(), List.of(), List.of());
    }

    /** Compact constructor with null-safe empty metadata. */
    public ChatPayload {
      Objects.requireNonNull(aiMessage, "aiMessage");
      metadata = metadata == null ? ChatMetadata.empty() : metadata;
      generationMetadata =
          generationMetadata == null ? GenerationMetadata.empty() : generationMetadata;
      additionalGenerations =
          additionalGenerations == null ? List.of() : List.copyOf(additionalGenerations);
      streamEvents = streamEvents == null ? List.of() : List.copyOf(streamEvents);
      streamChunks = streamChunks == null ? List.of() : List.copyOf(streamChunks);
    }
  }

  /** Assistant message content and tool-call requests. */
  record AiMessagePayload(
      String text,
      String thinking,
      List<ToolCall> toolExecutionRequests,
      Map<String, Object> attributes) {
    /** Compact constructor defensively copies tool calls and attributes. */
    public AiMessagePayload {
      toolExecutionRequests =
          toolExecutionRequests == null ? List.of() : List.copyOf(toolExecutionRequests);
      attributes = immutableMap(attributes);
    }
  }

  /** A framework-neutral tool execution request. */
  record ToolCall(String id, String type, String name, String arguments) {
    /** Constructor used by frameworks that do not expose a tool-call type. */
    public ToolCall(String id, String name, String arguments) {
      this(id, null, name, arguments);
    }

    /** Compact constructor validates the required tool name. */
    public ToolCall {
      Objects.requireNonNull(name, "name");
    }
  }

  /** Chat response metadata that should survive playback. */
  record ChatMetadata(
      String id,
      String modelName,
      TokenUsage tokenUsage,
      String finishReason,
      Map<String, Object> attributes,
      RateLimit rateLimit,
      List<PromptFilter> promptMetadata) {
    /** Constructor for the common cross-framework metadata fields. */
    public ChatMetadata(String id, String modelName, TokenUsage tokenUsage, String finishReason) {
      this(id, modelName, tokenUsage, finishReason, Map.of(), null, List.of());
    }

    /** Compact constructor defensively copies response attributes. */
    public ChatMetadata {
      attributes = immutableMap(attributes);
      promptMetadata = promptMetadata == null ? List.of() : List.copyOf(promptMetadata);
    }

    /** Shared empty metadata value. */
    public static ChatMetadata empty() {
      return new ChatMetadata(null, null, null, null, Map.of(), null, List.of());
    }
  }

  /** Token accounting for a chat response. */
  record TokenUsage(
      Integer inputTokenCount,
      Integer outputTokenCount,
      Integer totalTokenCount,
      Object nativeUsage) {
    /** Constructor for portable token counts without provider-native usage. */
    public TokenUsage(Integer inputTokenCount, Integer outputTokenCount, Integer totalTokenCount) {
      this(inputTokenCount, outputTokenCount, totalTokenCount, null);
    }
  }

  /** Per-generation metadata used by Spring AI responses and stream chunks. */
  record GenerationMetadata(
      String finishReason, Set<String> contentFilters, Map<String, Object> attributes) {
    /** Compact constructor defensively copies collections. */
    public GenerationMetadata {
      contentFilters = contentFilters == null ? Set.of() : Set.copyOf(contentFilters);
      attributes = immutableMap(attributes);
    }

    /** Shared empty generation metadata. */
    public static GenerationMetadata empty() {
      return new GenerationMetadata(null, Set.of(), Map.of());
    }
  }

  /** An additional chat generation beyond the primary result. */
  record ChatGenerationPayload(AiMessagePayload aiMessage, GenerationMetadata metadata) {
    /** Compact constructor with null-safe metadata. */
    public ChatGenerationPayload {
      Objects.requireNonNull(aiMessage, "aiMessage");
      metadata = metadata == null ? GenerationMetadata.empty() : metadata;
    }
  }

  /** A recorded callback emitted by a callback-oriented streaming model. */
  record StreamEvent(String type, String text, Integer index, ToolCall toolCall) {
    /** Compact constructor validates the event type. */
    public StreamEvent {
      Objects.requireNonNull(type, "type");
    }
  }

  /** Portable rate-limit snapshot from a chat response. Durations are stored in milliseconds. */
  record RateLimit(
      Long requestsLimit,
      Long requestsRemaining,
      Long requestsResetMillis,
      Long tokensLimit,
      Long tokensRemaining,
      Long tokensResetMillis) {}

  /** One Spring AI prompt-filter metadata entry. */
  record PromptFilter(int promptIndex, Object contentFilterMetadata) {}

  private static Map<String, Object> immutableMap(Map<String, Object> values) {
    if (values == null || values.isEmpty()) {
      return Map.of();
    }
    return Collections.unmodifiableMap(new LinkedHashMap<>(values));
  }
}
