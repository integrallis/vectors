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

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@link CassetteSerializer} implementation using Jackson's streaming API.
 *
 * <p>Produces the same compact JSON shape as the Avaje serializer in {@code
 * vectors-vcr-serde-avaje} so cassettes are interoperable across both implementations.
 */
public final class JacksonCassetteSerializer implements CassetteSerializer {

  private static final JsonFactory FACTORY = new JsonFactory();

  private static final String TYPE_EMBEDDING = "embedding";
  private static final String TYPE_BATCH_EMBEDDING = "batch_embedding";
  private static final String TYPE_CHAT = "chat";

  @Override
  public byte[] serialize(CassetteRecord record) {
    try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
        JsonGenerator g = FACTORY.createGenerator(baos)) {
      g.writeStartObject();
      if (record instanceof CassetteRecord.Embedding e) {
        g.writeStringField("type", TYPE_EMBEDDING);
        writeCommon(g, e.testId(), e.model(), e.timestamp());
        writeFloatArray(g, "embedding", e.embedding());
      } else if (record instanceof CassetteRecord.BatchEmbedding b) {
        g.writeStringField("type", TYPE_BATCH_EMBEDDING);
        writeCommon(g, b.testId(), b.model(), b.timestamp());
        g.writeArrayFieldStart("embeddings");
        for (float[] v : b.embeddings()) {
          writeFloatArrayBody(g, v);
        }
        g.writeEndArray();
      } else if (record instanceof CassetteRecord.Chat c) {
        g.writeStringField("type", TYPE_CHAT);
        writeCommon(g, c.testId(), c.model(), c.timestamp());
        g.writeStringField("prompt", c.prompt());
        writeChatPayload(g, c.response());
      } else {
        throw new IllegalArgumentException("unsupported record type: " + record.getClass());
      }
      g.writeEndObject();
      g.flush();
      return baos.toByteArray();
    } catch (IOException e) {
      throw new UncheckedIOException("Jackson cassette serialization failed", e);
    }
  }

  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    try (JsonParser p = FACTORY.createParser(new ByteArrayInputStream(bytes))) {
      if (p.nextToken() != JsonToken.START_OBJECT) {
        throw new IOException("expected START_OBJECT");
      }
      Map<String, Object> fields = new LinkedHashMap<>();
      while (p.nextToken() != JsonToken.END_OBJECT) {
        String name = p.currentName();
        JsonToken tok = p.nextToken();
        fields.put(name, readValue(p, tok));
      }
      String type = (String) fields.get("type");
      String testId = (String) fields.get("testId");
      String model = (String) fields.get("model");
      long timestamp = ((Number) fields.get("timestamp")).longValue();
      return switch (type) {
        case TYPE_EMBEDDING ->
            new CassetteRecord.Embedding(
                testId, model, timestamp, toFloatArray((List<?>) fields.get("embedding")));
        case TYPE_BATCH_EMBEDDING -> {
          List<?> arr = (List<?>) fields.get("embeddings");
          float[][] embeddings = new float[arr.size()][];
          for (int i = 0; i < arr.size(); i++) {
            embeddings[i] = toFloatArray((List<?>) arr.get(i));
          }
          yield new CassetteRecord.BatchEmbedding(testId, model, timestamp, embeddings);
        }
        case TYPE_CHAT -> {
          String prompt = (String) fields.get("prompt");
          yield new CassetteRecord.Chat(
              testId, model, timestamp, prompt, toChatPayload(fields.get("response")));
        }
        default -> throw new IOException("unknown cassette type: " + type);
      };
    } catch (IOException e) {
      throw new UncheckedIOException("Jackson cassette deserialization failed", e);
    }
  }

  private static void writeCommon(JsonGenerator g, String testId, String model, long timestamp)
      throws IOException {
    g.writeStringField("testId", testId);
    g.writeStringField("model", model);
    g.writeNumberField("timestamp", timestamp);
  }

  private static void writeFloatArray(JsonGenerator g, String fieldName, float[] values)
      throws IOException {
    g.writeArrayFieldStart(fieldName);
    for (float v : values) {
      g.writeNumber(v);
    }
    g.writeEndArray();
  }

  private static void writeFloatArrayBody(JsonGenerator g, float[] values) throws IOException {
    g.writeStartArray();
    for (float v : values) {
      g.writeNumber(v);
    }
    g.writeEndArray();
  }

  private static void writeChatPayload(JsonGenerator g, CassetteRecord.ChatPayload response)
      throws IOException {
    g.writeObjectFieldStart("response");
    writeAiMessage(g, response.aiMessage());
    writeChatMetadata(g, response.metadata());
    g.writeEndObject();
  }

  private static void writeAiMessage(JsonGenerator g, CassetteRecord.AiMessagePayload aiMessage)
      throws IOException {
    g.writeObjectFieldStart("aiMessage");
    writeNullableString(g, "text", aiMessage.text());
    writeNullableString(g, "thinking", aiMessage.thinking());
    g.writeArrayFieldStart("toolExecutionRequests");
    for (CassetteRecord.ToolCall tool : aiMessage.toolExecutionRequests()) {
      g.writeStartObject();
      writeNullableString(g, "id", tool.id());
      g.writeStringField("name", tool.name());
      writeNullableString(g, "arguments", tool.arguments());
      g.writeEndObject();
    }
    g.writeEndArray();
    g.writeObjectFieldStart("attributes");
    for (Map.Entry<String, Object> entry : aiMessage.attributes().entrySet()) {
      g.writeObjectField(entry.getKey(), entry.getValue());
    }
    g.writeEndObject();
    g.writeEndObject();
  }

  private static void writeChatMetadata(JsonGenerator g, CassetteRecord.ChatMetadata metadata)
      throws IOException {
    g.writeObjectFieldStart("metadata");
    writeNullableString(g, "id", metadata.id());
    writeNullableString(g, "modelName", metadata.modelName());
    if (metadata.tokenUsage() == null) {
      g.writeNullField("tokenUsage");
    } else {
      g.writeObjectFieldStart("tokenUsage");
      writeNullableNumber(g, "inputTokenCount", metadata.tokenUsage().inputTokenCount());
      writeNullableNumber(g, "outputTokenCount", metadata.tokenUsage().outputTokenCount());
      writeNullableNumber(g, "totalTokenCount", metadata.tokenUsage().totalTokenCount());
      g.writeEndObject();
    }
    writeNullableString(g, "finishReason", metadata.finishReason());
    g.writeEndObject();
  }

  private static void writeNullableString(JsonGenerator g, String fieldName, String value)
      throws IOException {
    if (value == null) {
      g.writeNullField(fieldName);
    } else {
      g.writeStringField(fieldName, value);
    }
  }

  private static void writeNullableNumber(JsonGenerator g, String fieldName, Integer value)
      throws IOException {
    if (value == null) {
      g.writeNullField(fieldName);
    } else {
      g.writeNumberField(fieldName, value);
    }
  }

  private static Object readValue(JsonParser p, JsonToken tok) throws IOException {
    return switch (tok) {
      case VALUE_STRING -> p.getValueAsString();
      case VALUE_NUMBER_INT -> p.getLongValue();
      case VALUE_NUMBER_FLOAT -> p.getDoubleValue();
      case VALUE_TRUE -> Boolean.TRUE;
      case VALUE_FALSE -> Boolean.FALSE;
      case VALUE_NULL -> null;
      case START_ARRAY -> {
        List<Object> list = new ArrayList<>();
        while (p.nextToken() != JsonToken.END_ARRAY) {
          list.add(readValue(p, p.currentToken()));
        }
        yield list;
      }
      case START_OBJECT -> {
        Map<String, Object> obj = new HashMap<>();
        while (p.nextToken() != JsonToken.END_OBJECT) {
          String n = p.currentName();
          obj.put(n, readValue(p, p.nextToken()));
        }
        yield obj;
      }
      default -> throw new IOException("unexpected token: " + tok);
    };
  }

  private static float[] toFloatArray(List<?> list) {
    float[] arr = new float[list.size()];
    for (int i = 0; i < list.size(); i++) {
      arr[i] = ((Number) list.get(i)).floatValue();
    }
    return arr;
  }

  private static CassetteRecord.ChatPayload toChatPayload(Object raw) throws IOException {
    Map<?, ?> map = requireMap(raw, "response");
    CassetteRecord.AiMessagePayload aiMessage = toAiMessage(map.get("aiMessage"));
    CassetteRecord.ChatMetadata metadata = toChatMetadata(map.get("metadata"));
    return new CassetteRecord.ChatPayload(aiMessage, metadata);
  }

  private static CassetteRecord.AiMessagePayload toAiMessage(Object raw) throws IOException {
    Map<?, ?> map = requireMap(raw, "response.aiMessage");
    List<CassetteRecord.ToolCall> tools = new ArrayList<>();
    Object rawTools = map.get("toolExecutionRequests");
    if (rawTools instanceof List<?> list) {
      for (Object rawTool : list) {
        Map<?, ?> tool = requireMap(rawTool, "toolExecutionRequests[]");
        tools.add(
            new CassetteRecord.ToolCall(
                asString(tool.get("id")),
                requireString(tool.get("name"), "toolExecutionRequests[].name"),
                asString(tool.get("arguments"))));
      }
    }
    return new CassetteRecord.AiMessagePayload(
        asString(map.get("text")),
        asString(map.get("thinking")),
        tools,
        toObjectMap(map.get("attributes")));
  }

  private static CassetteRecord.ChatMetadata toChatMetadata(Object raw) throws IOException {
    if (raw == null) {
      return CassetteRecord.ChatMetadata.empty();
    }
    Map<?, ?> map = requireMap(raw, "response.metadata");
    return new CassetteRecord.ChatMetadata(
        asString(map.get("id")),
        asString(map.get("modelName")),
        toTokenUsage(map.get("tokenUsage")),
        asString(map.get("finishReason")));
  }

  private static CassetteRecord.TokenUsage toTokenUsage(Object raw) throws IOException {
    if (raw == null) {
      return null;
    }
    Map<?, ?> map = requireMap(raw, "response.metadata.tokenUsage");
    return new CassetteRecord.TokenUsage(
        asInteger(map.get("inputTokenCount")),
        asInteger(map.get("outputTokenCount")),
        asInteger(map.get("totalTokenCount")));
  }

  private static Map<String, Object> toObjectMap(Object raw) throws IOException {
    if (raw == null) {
      return Map.of();
    }
    Map<?, ?> map = requireMap(raw, "attributes");
    Map<String, Object> out = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : map.entrySet()) {
      out.put(String.valueOf(entry.getKey()), entry.getValue());
    }
    return out;
  }

  private static Map<?, ?> requireMap(Object raw, String field) throws IOException {
    if (raw instanceof Map<?, ?> map) {
      return map;
    }
    throw new IOException("expected object field: " + field);
  }

  private static String requireString(Object raw, String field) throws IOException {
    if (raw instanceof String s) {
      return s;
    }
    throw new IOException("expected string field: " + field);
  }

  private static String asString(Object raw) {
    return raw instanceof String s ? s : null;
  }

  private static Integer asInteger(Object raw) {
    return raw instanceof Number n ? n.intValue() : null;
  }
}
