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
package com.integrallis.vectors.vcr.semantic;

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Test-only {@link CassetteSerializer} using raw Java {@code DataOutput}. Avoids depending on any
 * JSON library in this module's unit tests.
 */
final class TestSerializer implements CassetteSerializer {

  private static final byte TYPE_EMBEDDING = 1;
  private static final byte TYPE_BATCH_EMBEDDING = 2;
  private static final byte TYPE_CHAT = 3;

  @Override
  public byte[] serialize(CassetteRecord record) {
    try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos)) {
      if (record instanceof CassetteRecord.Embedding e) {
        out.writeByte(TYPE_EMBEDDING);
        writeCommon(out, e.testId(), e.model(), e.timestamp());
        writeFloatArray(out, e.embedding());
      } else if (record instanceof CassetteRecord.BatchEmbedding b) {
        out.writeByte(TYPE_BATCH_EMBEDDING);
        writeCommon(out, b.testId(), b.model(), b.timestamp());
        out.writeInt(b.embeddings().length);
        for (float[] v : b.embeddings()) {
          writeFloatArray(out, v);
        }
      } else if (record instanceof CassetteRecord.Chat c) {
        out.writeByte(TYPE_CHAT);
        writeCommon(out, c.testId(), c.model(), c.timestamp());
        out.writeUTF(c.prompt());
        writeChatPayload(out, c.response());
      } else {
        throw new IllegalArgumentException("unknown record type: " + record.getClass());
      }
      return baos.toByteArray();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
      byte type = in.readByte();
      String testId = in.readUTF();
      String model = in.readUTF();
      long timestamp = in.readLong();
      switch (type) {
        case TYPE_EMBEDDING -> {
          return new CassetteRecord.Embedding(testId, model, timestamp, readFloatArray(in));
        }
        case TYPE_BATCH_EMBEDDING -> {
          int n = in.readInt();
          float[][] embeddings = new float[n][];
          for (int i = 0; i < n; i++) {
            embeddings[i] = readFloatArray(in);
          }
          return new CassetteRecord.BatchEmbedding(testId, model, timestamp, embeddings);
        }
        case TYPE_CHAT -> {
          String prompt = in.readUTF();
          return new CassetteRecord.Chat(testId, model, timestamp, prompt, readChatPayload(in));
        }
        default -> throw new IllegalArgumentException("unknown type byte: " + type);
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static void writeCommon(DataOutputStream out, String testId, String model, long timestamp)
      throws IOException {
    out.writeUTF(testId);
    out.writeUTF(model);
    out.writeLong(timestamp);
  }

  private static void writeFloatArray(DataOutputStream out, float[] values) throws IOException {
    out.writeInt(values.length);
    for (float v : values) {
      out.writeFloat(v);
    }
  }

  private static void writeChatPayload(DataOutputStream out, CassetteRecord.ChatPayload response)
      throws IOException {
    CassetteRecord.AiMessagePayload ai = response.aiMessage();
    writeNullableUtf(out, ai.text());
    writeNullableUtf(out, ai.thinking());
    out.writeInt(ai.toolExecutionRequests().size());
    for (CassetteRecord.ToolCall tool : ai.toolExecutionRequests()) {
      writeNullableUtf(out, tool.id());
      out.writeUTF(tool.name());
      writeNullableUtf(out, tool.arguments());
    }
    out.writeInt(ai.attributes().size());
    for (Map.Entry<String, Object> entry : ai.attributes().entrySet()) {
      out.writeUTF(entry.getKey());
      writeNullableUtf(out, entry.getValue() == null ? null : String.valueOf(entry.getValue()));
    }
    CassetteRecord.ChatMetadata metadata = response.metadata();
    writeNullableUtf(out, metadata.id());
    writeNullableUtf(out, metadata.modelName());
    out.writeBoolean(metadata.tokenUsage() != null);
    if (metadata.tokenUsage() != null) {
      writeNullableInt(out, metadata.tokenUsage().inputTokenCount());
      writeNullableInt(out, metadata.tokenUsage().outputTokenCount());
      writeNullableInt(out, metadata.tokenUsage().totalTokenCount());
    }
    writeNullableUtf(out, metadata.finishReason());
  }

  private static float[] readFloatArray(DataInputStream in) throws IOException {
    int n = in.readInt();
    float[] arr = new float[n];
    for (int i = 0; i < n; i++) {
      arr[i] = in.readFloat();
    }
    return arr;
  }

  private static CassetteRecord.ChatPayload readChatPayload(DataInputStream in) throws IOException {
    String text = readNullableUtf(in);
    String thinking = readNullableUtf(in);
    int toolCount = in.readInt();
    CassetteRecord.ToolCall[] tools = new CassetteRecord.ToolCall[toolCount];
    for (int i = 0; i < toolCount; i++) {
      tools[i] =
          new CassetteRecord.ToolCall(readNullableUtf(in), in.readUTF(), readNullableUtf(in));
    }
    int attributeCount = in.readInt();
    Map<String, Object> attributes = new HashMap<>(attributeCount);
    for (int i = 0; i < attributeCount; i++) {
      attributes.put(in.readUTF(), readNullableUtf(in));
    }
    String id = readNullableUtf(in);
    String modelName = readNullableUtf(in);
    CassetteRecord.TokenUsage tokenUsage = null;
    if (in.readBoolean()) {
      tokenUsage =
          new CassetteRecord.TokenUsage(
              readNullableInt(in), readNullableInt(in), readNullableInt(in));
    }
    String finishReason = readNullableUtf(in);
    return new CassetteRecord.ChatPayload(
        new CassetteRecord.AiMessagePayload(text, thinking, List.of(tools), attributes),
        new CassetteRecord.ChatMetadata(id, modelName, tokenUsage, finishReason));
  }

  private static void writeNullableUtf(DataOutputStream out, String value) throws IOException {
    out.writeBoolean(value != null);
    if (value != null) {
      out.writeUTF(value);
    }
  }

  private static String readNullableUtf(DataInputStream in) throws IOException {
    return in.readBoolean() ? in.readUTF() : null;
  }

  private static void writeNullableInt(DataOutputStream out, Integer value) throws IOException {
    out.writeBoolean(value != null);
    if (value != null) {
      out.writeInt(value);
    }
  }

  private static Integer readNullableInt(DataInputStream in) throws IOException {
    return in.readBoolean() ? in.readInt() : null;
  }
}
