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
package com.integrallis.vectors.vcr.serde.avaje;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class AvajeCassetteSerializerTest {

  private final CassetteSerializer serializer = new AvajeCassetteSerializer();

  @Test
  void roundTripEmbedding() {
    CassetteRecord.Embedding in =
        new CassetteRecord.Embedding("T:1", "m", 42L, new float[] {1f, -2.5f, 3e-3f});
    CassetteRecord.Embedding back =
        (CassetteRecord.Embedding) serializer.deserialize(serializer.serialize(in));
    assertEquals("T:1", back.testId());
    assertEquals("m", back.model());
    assertEquals(42L, back.timestamp());
    assertArrayEquals(in.embedding(), back.embedding());
  }

  @Test
  void roundTripBatchEmbedding() {
    float[][] batch = {
      {1f, 2f},
      {3f, 4f, 5f}
    };
    CassetteRecord.BatchEmbedding in = new CassetteRecord.BatchEmbedding("T:b", "m", 0L, batch);
    CassetteRecord.BatchEmbedding back =
        (CassetteRecord.BatchEmbedding) serializer.deserialize(serializer.serialize(in));
    assertEquals(2, back.embeddings().length);
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
            new CassetteRecord.ChatPayload(
                new CassetteRecord.AiMessagePayload(
                    "world",
                    "thinking",
                    List.of(new CassetteRecord.ToolCall("call-1", "search", "{\"q\":\"x\"}")),
                    Map.of("source", "unit")),
                new CassetteRecord.ChatMetadata(
                    "resp-1", "gpt-4", new CassetteRecord.TokenUsage(2, 3, 5), "TOOL_EXECUTION")));
    CassetteRecord.Chat back =
        (CassetteRecord.Chat) serializer.deserialize(serializer.serialize(in));
    assertEquals("hello", back.prompt());
    assertEquals("world", back.response().aiMessage().text());
    assertEquals("search", back.response().aiMessage().toolExecutionRequests().getFirst().name());
    assertEquals(5, back.response().metadata().tokenUsage().totalTokenCount());
    assertEquals("TOOL_EXECUTION", back.response().metadata().finishReason());
  }

  @Test
  void serviceLoaderFindsSerializer() {
    CassetteSerializer loaded = CassetteSerializer.load();
    assertInstanceOf(AvajeCassetteSerializer.class, loaded);
  }
}
