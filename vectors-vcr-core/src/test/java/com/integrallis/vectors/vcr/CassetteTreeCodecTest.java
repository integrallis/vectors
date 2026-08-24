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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CassetteTreeCodecTest {

  @Test
  void roundTripsEveryCassetteShapeThroughOneCanonicalTree() {
    CassetteRecord.Embedding embedding =
        new CassetteRecord.Embedding("T:e", "embed", 1L, new float[] {1f, -2.5f}, "sha256:e");
    CassetteRecord.BatchEmbedding batch =
        new CassetteRecord.BatchEmbedding(
            "T:b", "embed", 2L, new float[][] {{1f, 2f}, {-3f}}, "sha256:b");
    CassetteRecord.Chat chat =
        new CassetteRecord.Chat("T:c", "chat", 3L, "hello", richPayload(), "sha256:c");

    Map<String, Object> embeddingTree = CassetteTreeCodec.toTree(embedding);
    Map<String, Object> batchTree = CassetteTreeCodec.toTree(batch);
    Map<String, Object> chatTree = CassetteTreeCodec.toTree(chat);

    assertThat(embeddingTree)
        .containsEntry("type", "embedding")
        .containsEntry("requestSignature", "sha256:e");
    assertThat(batchTree).containsEntry("type", "batch_embedding");
    assertThat(chatTree)
        .containsEntry("type", "chat")
        .containsEntry("prompt", "hello")
        .containsKey("response");

    CassetteRecord.Embedding decodedEmbedding =
        (CassetteRecord.Embedding) CassetteTreeCodec.fromTree(embeddingTree);
    CassetteRecord.BatchEmbedding decodedBatch =
        (CassetteRecord.BatchEmbedding) CassetteTreeCodec.fromTree(batchTree);
    CassetteRecord.Chat decodedChat = (CassetteRecord.Chat) CassetteTreeCodec.fromTree(chatTree);

    assertThat(decodedEmbedding.testId()).isEqualTo(embedding.testId());
    assertThat(decodedEmbedding.embedding()).containsExactly(embedding.embedding());
    assertThat(decodedBatch.embeddings().length).isEqualTo(2);
    assertThat(decodedBatch.embeddings()[0]).containsExactly(batch.embeddings()[0]);
    assertThat(decodedBatch.embeddings()[1]).containsExactly(batch.embeddings()[1]);
    assertThat(decodedChat).isEqualTo(chat);
  }

  @Test
  void rejectsUnknownAndMalformedTreesAtTheSharedBoundary() {
    assertThatThrownBy(() -> CassetteTreeCodec.fromTree(Map.of("type", "unknown")))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("unknown cassette type");
    assertThatThrownBy(() -> CassetteTreeCodec.fromTree(Map.of("type", "chat")))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static CassetteRecord.ChatPayload richPayload() {
    CassetteRecord.AiMessagePayload primary =
        new CassetteRecord.AiMessagePayload(
            "world",
            "thinking",
            List.of(new CassetteRecord.ToolCall("call-1", "function", "search", "{}")),
            Map.of("message", "primary"));
    CassetteRecord.ChatMetadata metadata =
        new CassetteRecord.ChatMetadata(
            "resp-1",
            "chat",
            new CassetteRecord.TokenUsage(5, 6, 11, Map.of("cached", true)),
            "STOP",
            Map.of("fingerprint", "fp-1"),
            new CassetteRecord.RateLimit(10L, 9L, 1000L, 100L, 90L, 2000L),
            List.of(new CassetteRecord.PromptFilter(0, Map.of("safe", true))));
    CassetteRecord.GenerationMetadata generation =
        new CassetteRecord.GenerationMetadata("STOP", Set.of("safe"), Map.of("generation", "g-1"));
    CassetteRecord.ChatPayload chunk =
        new CassetteRecord.ChatPayload(
            new CassetteRecord.AiMessagePayload("wor", null, List.of(), Map.of()), metadata);
    return new CassetteRecord.ChatPayload(
        primary,
        metadata,
        generation,
        List.of(
            new CassetteRecord.ChatGenerationPayload(
                new CassetteRecord.AiMessagePayload("alternate", null, List.of(), Map.of()),
                generation)),
        List.of(
            new CassetteRecord.StreamEvent("partial_response", "wor", null, null),
            new CassetteRecord.StreamEvent(
                "complete_tool_call",
                null,
                0,
                new CassetteRecord.ToolCall("call-1", "function", "search", "{}"))),
        List.of(chunk));
  }
}
