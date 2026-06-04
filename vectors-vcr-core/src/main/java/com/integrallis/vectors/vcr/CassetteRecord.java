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

import java.util.List;
import java.util.Map;
import java.util.Objects;

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

  /** A single embedding vector. */
  record Embedding(String testId, String model, long timestamp, float[] embedding)
      implements CassetteRecord {
    /** Compact constructor with null/shape checks. */
    public Embedding {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(embedding, "embedding");
    }
  }

  /** A batch of embedding vectors produced by a single batch call. */
  record BatchEmbedding(String testId, String model, long timestamp, float[][] embeddings)
      implements CassetteRecord {
    /** Compact constructor with null/shape checks. */
    public BatchEmbedding {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(embeddings, "embeddings");
    }
  }

  /** A chat exchange with enough structured response data for lossless framework playback. */
  record Chat(String testId, String model, long timestamp, String prompt, ChatPayload response)
      implements CassetteRecord {
    /** Compact constructor validates non-null prompt and response payloads. */
    public Chat {
      Objects.requireNonNull(testId, "testId");
      Objects.requireNonNull(model, "model");
      Objects.requireNonNull(prompt, "prompt");
      Objects.requireNonNull(response, "response");
    }
  }

  /** Structured chat response payload. */
  record ChatPayload(AiMessagePayload aiMessage, ChatMetadata metadata) {
    /** Compact constructor with null-safe empty metadata. */
    public ChatPayload {
      Objects.requireNonNull(aiMessage, "aiMessage");
      metadata = metadata == null ? ChatMetadata.empty() : metadata;
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
      attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }
  }

  /** A framework-neutral tool execution request. */
  record ToolCall(String id, String name, String arguments) {
    /** Compact constructor validates the required tool name. */
    public ToolCall {
      Objects.requireNonNull(name, "name");
    }
  }

  /** Chat response metadata that should survive playback. */
  record ChatMetadata(String id, String modelName, TokenUsage tokenUsage, String finishReason) {
    /** Shared empty metadata value. */
    public static ChatMetadata empty() {
      return new ChatMetadata(null, null, null, null);
    }
  }

  /** Token accounting for a chat response. */
  record TokenUsage(Integer inputTokenCount, Integer outputTokenCount, Integer totalTokenCount) {}
}
