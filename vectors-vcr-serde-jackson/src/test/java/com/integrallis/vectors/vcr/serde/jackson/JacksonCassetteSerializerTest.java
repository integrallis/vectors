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
package com.integrallis.vectors.vcr.serde.jackson;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import com.integrallis.vectors.vcr.serde.avaje.AvajeCassetteSerializer;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JacksonCassetteSerializerTest {

  private final CassetteSerializer serializer = new JacksonCassetteSerializer();

  @Test
  void roundTripEmbedding() {
    CassetteRecord.Embedding in =
        new CassetteRecord.Embedding("T:1", "m", 42L, new float[] {1f, -2.5f, 3e-3f}, "sha256:one");
    CassetteRecord out = serializer.deserialize(serializer.serialize(in));
    assertInstanceOf(CassetteRecord.Embedding.class, out);
    CassetteRecord.Embedding back = (CassetteRecord.Embedding) out;
    assertEquals("T:1", back.testId());
    assertEquals("m", back.model());
    assertEquals(42L, back.timestamp());
    assertEquals("sha256:one", back.requestSignature());
    assertArrayEquals(in.embedding(), back.embedding());
  }

  @Test
  void roundTripBatchEmbedding() {
    float[][] batch = {
      {1f, 2f},
      {3f, 4f, 5f}
    };
    CassetteRecord.BatchEmbedding in =
        new CassetteRecord.BatchEmbedding("T:b", "m", 0L, batch, "sha256:batch");
    CassetteRecord.BatchEmbedding back =
        (CassetteRecord.BatchEmbedding) serializer.deserialize(serializer.serialize(in));
    assertEquals(2, back.embeddings().length);
    assertEquals("sha256:batch", back.requestSignature());
    assertArrayEquals(batch[0], back.embeddings()[0]);
    assertArrayEquals(batch[1], back.embeddings()[1]);
  }

  @Test
  void roundTripChat() {
    CassetteRecord.Chat in =
        new CassetteRecord.Chat(
            "T:c",
            "gpt",
            5L,
            "hello",
            chatPayload(
                "world", List.of(new CassetteRecord.ToolCall("call-1", "search", "{\"q\":\"x\"}"))),
            "sha256:chat");
    CassetteRecord.Chat back =
        (CassetteRecord.Chat) serializer.deserialize(serializer.serialize(in));
    assertEquals("hello", back.prompt());
    assertEquals("world", back.response().aiMessage().text());
    assertEquals("search", back.response().aiMessage().toolExecutionRequests().getFirst().name());
    assertEquals(11, back.response().metadata().tokenUsage().totalTokenCount());
    assertEquals("STOP", back.response().metadata().finishReason());
    assertEquals("sha256:chat", back.requestSignature());
  }

  @Test
  void contentTypeIsJson() {
    assertEquals("application/json", serializer.contentType());
  }

  @Test
  void serviceLoaderFindsSerializer() {
    boolean found =
        ServiceLoader.load(CassetteSerializer.class).stream()
            .anyMatch(provider -> provider.type().equals(JacksonCassetteSerializer.class));
    assertTrue(found);
  }

  @Test
  void avajeSerializedEmbeddingReadsWithJacksonAndReverse() {
    CassetteRecord.Embedding in =
        new CassetteRecord.Embedding("T:interop", "embedder", 123L, new float[] {1f, -2f, 0.5f});

    assertSameRecord(in, serializer.deserialize(new AvajeCassetteSerializer().serialize(in)));
    assertSameRecord(in, new AvajeCassetteSerializer().deserialize(serializer.serialize(in)));
  }

  @Test
  void avajeSerializedBatchEmbeddingReadsWithJacksonAndReverse() {
    CassetteRecord.BatchEmbedding in =
        new CassetteRecord.BatchEmbedding(
            "T:batch",
            "embedder",
            124L,
            new float[][] {
              {1f, 2f},
              {-3f, 4.25f}
            });

    assertSameRecord(in, serializer.deserialize(new AvajeCassetteSerializer().serialize(in)));
    assertSameRecord(in, new AvajeCassetteSerializer().deserialize(serializer.serialize(in)));
  }

  @Test
  void avajeSerializedChatReadsWithJacksonAndReverse() {
    CassetteRecord.Chat in =
        new CassetteRecord.Chat(
            "T:chat", "chat-model", 125L, "hello", richChatPayload(), "sha256:interop");

    assertSameRecord(in, serializer.deserialize(new AvajeCassetteSerializer().serialize(in)));
    assertSameRecord(in, new AvajeCassetteSerializer().deserialize(serializer.serialize(in)));
  }

  private static void assertSameRecord(CassetteRecord expected, CassetteRecord actual) {
    assertEquals(expected.testId(), actual.testId());
    assertEquals(expected.model(), actual.model());
    assertEquals(expected.timestamp(), actual.timestamp());
    assertEquals(expected.requestSignature(), actual.requestSignature());
    if (expected instanceof CassetteRecord.Embedding e) {
      CassetteRecord.Embedding a = assertInstanceOf(CassetteRecord.Embedding.class, actual);
      assertArrayEquals(e.embedding(), a.embedding());
    } else if (expected instanceof CassetteRecord.BatchEmbedding e) {
      CassetteRecord.BatchEmbedding a =
          assertInstanceOf(CassetteRecord.BatchEmbedding.class, actual);
      assertEquals(e.embeddings().length, a.embeddings().length);
      for (int i = 0; i < e.embeddings().length; i++) {
        assertArrayEquals(e.embeddings()[i], a.embeddings()[i]);
      }
    } else if (expected instanceof CassetteRecord.Chat e) {
      CassetteRecord.Chat a = assertInstanceOf(CassetteRecord.Chat.class, actual);
      assertEquals(e.prompt(), a.prompt());
      assertEquals(e.response(), a.response());
    } else {
      throw new AssertionError("unsupported record type: " + expected.getClass());
    }
  }

  private static CassetteRecord.ChatPayload chatPayload(
      String text, List<CassetteRecord.ToolCall> tools) {
    return new CassetteRecord.ChatPayload(
        new CassetteRecord.AiMessagePayload(
            text, "chain-of-thought-redacted", tools, Map.of("source", "unit")),
        new CassetteRecord.ChatMetadata(
            "resp-1", "chat-model", new CassetteRecord.TokenUsage(5, 6, 11), "STOP"));
  }

  private static CassetteRecord.ChatPayload richChatPayload() {
    CassetteRecord.AiMessagePayload primary =
        new CassetteRecord.AiMessagePayload(
            "world",
            "thinking",
            List.of(new CassetteRecord.ToolCall("call-1", "function", "search", "{}")),
            Map.of("message", "primary"));
    CassetteRecord.ChatMetadata metadata =
        new CassetteRecord.ChatMetadata(
            "resp-1",
            "chat-model",
            new CassetteRecord.TokenUsage(5, 6, 11, Map.of("cached", "yes")),
            "STOP",
            Map.of("fingerprint", "fp-1"),
            new CassetteRecord.RateLimit(10L, 9L, 1000L, 100L, 90L, 2000L),
            List.of(new CassetteRecord.PromptFilter(0, Map.of("safe", "true"))));
    CassetteRecord.GenerationMetadata generation =
        new CassetteRecord.GenerationMetadata(
            "STOP", java.util.Set.of("safe"), Map.of("generation", "g-1"));
    CassetteRecord.ChatGenerationPayload additional =
        new CassetteRecord.ChatGenerationPayload(
            new CassetteRecord.AiMessagePayload("alternate", null, List.of(), Map.of()),
            generation);
    CassetteRecord.StreamEvent event =
        new CassetteRecord.StreamEvent("partial_response", "wor", null, null);
    CassetteRecord.ChatPayload chunk =
        new CassetteRecord.ChatPayload(
            new CassetteRecord.AiMessagePayload("wor", null, List.of(), Map.of()), metadata);
    return new CassetteRecord.ChatPayload(
        primary, metadata, generation, List.of(additional), List.of(event), List.of(chunk));
  }
}
