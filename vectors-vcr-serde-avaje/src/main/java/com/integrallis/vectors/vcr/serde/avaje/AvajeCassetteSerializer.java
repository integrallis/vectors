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

import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteSerializer;
import io.avaje.jsonb.JsonType;
import io.avaje.jsonb.Jsonb;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@link CassetteSerializer} backed by Avaje {@code Jsonb}.
 *
 * <p>Serializes each {@link CassetteRecord} to a plain {@code Map} tree so no annotation processor
 * is required. The JSON output shape matches the Jackson implementation in {@code
 * vectors-vcr-serde-jackson} for cross-serializer interoperability.
 */
public final class AvajeCassetteSerializer implements CassetteSerializer {

  private static final String TYPE_EMBEDDING = "embedding";
  private static final String TYPE_BATCH_EMBEDDING = "batch_embedding";
  private static final String TYPE_CHAT = "chat";

  private final Jsonb jsonb = Jsonb.builder().build();
  private final JsonType<Object> anyType = jsonb.type(Object.class);

  @Override
  public byte[] serialize(CassetteRecord record) {
    Map<String, Object> tree = toTree(record);
    return anyType.toJsonBytes(tree);
  }

  @Override
  public CassetteRecord deserialize(byte[] bytes) {
    Object parsed = anyType.fromJson(bytes);
    if (!(parsed instanceof Map<?, ?> map)) {
      throw new IllegalArgumentException("expected JSON object at top level");
    }
    return fromTree(map);
  }

  private static Map<String, Object> toTree(CassetteRecord record) {
    Map<String, Object> tree = new LinkedHashMap<>();
    if (record instanceof CassetteRecord.Embedding e) {
      tree.put("type", TYPE_EMBEDDING);
      tree.put("testId", e.testId());
      tree.put("model", e.model());
      tree.put("timestamp", e.timestamp());
      tree.put("embedding", asList(e.embedding()));
    } else if (record instanceof CassetteRecord.BatchEmbedding b) {
      tree.put("type", TYPE_BATCH_EMBEDDING);
      tree.put("testId", b.testId());
      tree.put("model", b.model());
      tree.put("timestamp", b.timestamp());
      List<List<Double>> outer = new ArrayList<>(b.embeddings().length);
      for (float[] v : b.embeddings()) {
        outer.add(asList(v));
      }
      tree.put("embeddings", outer);
    } else if (record instanceof CassetteRecord.Chat c) {
      tree.put("type", TYPE_CHAT);
      tree.put("testId", c.testId());
      tree.put("model", c.model());
      tree.put("timestamp", c.timestamp());
      tree.put("prompt", c.prompt());
      tree.put("response", chatPayloadToTree(c.response()));
    } else {
      throw new IllegalArgumentException("unsupported record type: " + record.getClass());
    }
    return tree;
  }

  private static CassetteRecord fromTree(Map<?, ?> map) {
    String type = (String) map.get("type");
    String testId = (String) map.get("testId");
    String model = (String) map.get("model");
    long timestamp = ((Number) map.get("timestamp")).longValue();
    return switch (type) {
      case TYPE_EMBEDDING ->
          new CassetteRecord.Embedding(
              testId, model, timestamp, toFloatArray((List<?>) map.get("embedding")));
      case TYPE_BATCH_EMBEDDING -> {
        List<?> outer = (List<?>) map.get("embeddings");
        float[][] embeddings = new float[outer.size()][];
        for (int i = 0; i < outer.size(); i++) {
          embeddings[i] = toFloatArray((List<?>) outer.get(i));
        }
        yield new CassetteRecord.BatchEmbedding(testId, model, timestamp, embeddings);
      }
      case TYPE_CHAT -> {
        String prompt = (String) map.get("prompt");
        yield new CassetteRecord.Chat(
            testId, model, timestamp, prompt, toChatPayload(map.get("response")));
      }
      default -> throw new IllegalArgumentException("unknown cassette type: " + type);
    };
  }

  private static List<Double> asList(float[] values) {
    List<Double> out = new ArrayList<>(values.length);
    for (float v : values) {
      out.add((double) v);
    }
    return out;
  }

  private static Map<String, Object> chatPayloadToTree(CassetteRecord.ChatPayload response) {
    Map<String, Object> tree = new LinkedHashMap<>();
    CassetteRecord.AiMessagePayload aiMessage = response.aiMessage();
    Map<String, Object> ai = new LinkedHashMap<>();
    ai.put("text", aiMessage.text());
    ai.put("thinking", aiMessage.thinking());
    List<Map<String, Object>> tools = new ArrayList<>(aiMessage.toolExecutionRequests().size());
    for (CassetteRecord.ToolCall tool : aiMessage.toolExecutionRequests()) {
      Map<String, Object> item = new LinkedHashMap<>();
      item.put("id", tool.id());
      item.put("name", tool.name());
      item.put("arguments", tool.arguments());
      tools.add(item);
    }
    ai.put("toolExecutionRequests", tools);
    ai.put("attributes", new LinkedHashMap<>(aiMessage.attributes()));
    tree.put("aiMessage", ai);

    CassetteRecord.ChatMetadata metadata = response.metadata();
    Map<String, Object> md = new LinkedHashMap<>();
    md.put("id", metadata.id());
    md.put("modelName", metadata.modelName());
    md.put("finishReason", metadata.finishReason());
    if (metadata.tokenUsage() == null) {
      md.put("tokenUsage", null);
    } else {
      Map<String, Object> usage = new LinkedHashMap<>();
      usage.put("inputTokenCount", metadata.tokenUsage().inputTokenCount());
      usage.put("outputTokenCount", metadata.tokenUsage().outputTokenCount());
      usage.put("totalTokenCount", metadata.tokenUsage().totalTokenCount());
      md.put("tokenUsage", usage);
    }
    tree.put("metadata", md);
    return tree;
  }

  private static float[] toFloatArray(List<?> list) {
    float[] arr = new float[list.size()];
    for (int i = 0; i < list.size(); i++) {
      arr[i] = ((Number) list.get(i)).floatValue();
    }
    return arr;
  }

  private static CassetteRecord.ChatPayload toChatPayload(Object raw) {
    Map<?, ?> map = requireMap(raw, "response");
    return new CassetteRecord.ChatPayload(
        toAiMessage(map.get("aiMessage")), toChatMetadata(map.get("metadata")));
  }

  private static CassetteRecord.AiMessagePayload toAiMessage(Object raw) {
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

  private static CassetteRecord.ChatMetadata toChatMetadata(Object raw) {
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

  private static CassetteRecord.TokenUsage toTokenUsage(Object raw) {
    if (raw == null) {
      return null;
    }
    Map<?, ?> map = requireMap(raw, "response.metadata.tokenUsage");
    return new CassetteRecord.TokenUsage(
        asInteger(map.get("inputTokenCount")),
        asInteger(map.get("outputTokenCount")),
        asInteger(map.get("totalTokenCount")));
  }

  private static Map<String, Object> toObjectMap(Object raw) {
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

  private static Map<?, ?> requireMap(Object raw, String field) {
    if (raw instanceof Map<?, ?> map) {
      return map;
    }
    throw new IllegalArgumentException("expected object field: " + field);
  }

  private static String requireString(Object raw, String field) {
    if (raw instanceof String s) {
      return s;
    }
    throw new IllegalArgumentException("expected string field: " + field);
  }

  private static String asString(Object raw) {
    return raw instanceof String s ? s : null;
  }

  private static Integer asInteger(Object raw) {
    return raw instanceof Number n ? n.intValue() : null;
  }
}
