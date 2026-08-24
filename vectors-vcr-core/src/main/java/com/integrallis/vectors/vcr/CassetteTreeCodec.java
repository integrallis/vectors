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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Maps cassette records to and from their canonical JSON-compatible object tree.
 *
 * <p>Serializer modules own only the JSON engine boundary. Keeping the cassette shape here ensures
 * that Avaje and Jackson encode the same fields as the record model evolves.
 */
public final class CassetteTreeCodec {

  private static final String TYPE_EMBEDDING = "embedding";
  private static final String TYPE_BATCH_EMBEDDING = "batch_embedding";
  private static final String TYPE_CHAT = "chat";

  private CassetteTreeCodec() {}

  /**
   * Converts a cassette record into maps, lists, strings, numbers, booleans, and nulls.
   *
   * @param record cassette record to encode
   * @return mutable canonical object tree
   * @throws IllegalArgumentException if the record implementation is unsupported
   */
  public static Map<String, Object> toTree(CassetteRecord record) {
    Map<String, Object> tree = new LinkedHashMap<>();
    if (record instanceof CassetteRecord.Embedding embedding) {
      putCommon(tree, TYPE_EMBEDDING, embedding);
      tree.put("embedding", asList(embedding.embedding()));
    } else if (record instanceof CassetteRecord.BatchEmbedding batch) {
      putCommon(tree, TYPE_BATCH_EMBEDDING, batch);
      List<List<Double>> embeddings = new ArrayList<>(batch.embeddings().length);
      for (float[] vector : batch.embeddings()) {
        embeddings.add(asList(vector));
      }
      tree.put("embeddings", embeddings);
    } else if (record instanceof CassetteRecord.Chat chat) {
      putCommon(tree, TYPE_CHAT, chat);
      tree.put("prompt", chat.prompt());
      tree.put("response", chatPayloadToTree(chat.response()));
    } else {
      throw new IllegalArgumentException("unsupported record type: " + record.getClass());
    }
    return tree;
  }

  /**
   * Reconstructs a cassette record from a JSON-compatible object tree.
   *
   * @param tree decoded object tree
   * @return reconstructed cassette record
   * @throws IllegalArgumentException if required fields are missing or malformed
   */
  public static CassetteRecord fromTree(Map<?, ?> tree) {
    String type = requireString(tree.get("type"), "type");
    if (!TYPE_EMBEDDING.equals(type)
        && !TYPE_BATCH_EMBEDDING.equals(type)
        && !TYPE_CHAT.equals(type)) {
      throw new IllegalArgumentException("unknown cassette type: " + type);
    }

    String testId = requireString(tree.get("testId"), "testId");
    String model = requireString(tree.get("model"), "model");
    long timestamp = requireNumber(tree.get("timestamp"), "timestamp").longValue();
    String requestSignature = asString(tree.get("requestSignature"));
    return switch (type) {
      case TYPE_EMBEDDING ->
          new CassetteRecord.Embedding(
              testId,
              model,
              timestamp,
              toFloatArray(requireList(tree.get("embedding"), "embedding")),
              requestSignature);
      case TYPE_BATCH_EMBEDDING -> {
        List<?> vectors = requireList(tree.get("embeddings"), "embeddings");
        float[][] embeddings = new float[vectors.size()][];
        for (int i = 0; i < vectors.size(); i++) {
          embeddings[i] = toFloatArray(requireList(vectors.get(i), "embeddings[" + i + "]"));
        }
        yield new CassetteRecord.BatchEmbedding(
            testId, model, timestamp, embeddings, requestSignature);
      }
      case TYPE_CHAT ->
          new CassetteRecord.Chat(
              testId,
              model,
              timestamp,
              requireString(tree.get("prompt"), "prompt"),
              toChatPayload(tree.get("response")),
              requestSignature);
      default -> throw new IllegalStateException("validated cassette type changed: " + type);
    };
  }

  private static void putCommon(Map<String, Object> tree, String type, CassetteRecord record) {
    tree.put("type", type);
    tree.put("testId", record.testId());
    tree.put("model", record.model());
    tree.put("timestamp", record.timestamp());
    tree.put("requestSignature", record.requestSignature());
  }

  private static List<Double> asList(float[] values) {
    List<Double> result = new ArrayList<>(values.length);
    for (float value : values) {
      result.add((double) value);
    }
    return result;
  }

  private static Map<String, Object> chatPayloadToTree(CassetteRecord.ChatPayload response) {
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("aiMessage", aiMessageToTree(response.aiMessage()));
    tree.put("generationMetadata", generationMetadataToTree(response.generationMetadata()));

    List<Map<String, Object>> additional = new ArrayList<>(response.additionalGenerations().size());
    for (CassetteRecord.ChatGenerationPayload generation : response.additionalGenerations()) {
      Map<String, Object> item = new LinkedHashMap<>();
      item.put("aiMessage", aiMessageToTree(generation.aiMessage()));
      item.put("metadata", generationMetadataToTree(generation.metadata()));
      additional.add(item);
    }
    tree.put("additionalGenerations", additional);

    List<Map<String, Object>> events = new ArrayList<>(response.streamEvents().size());
    for (CassetteRecord.StreamEvent event : response.streamEvents()) {
      Map<String, Object> item = new LinkedHashMap<>();
      item.put("type", event.type());
      item.put("text", event.text());
      item.put("index", event.index());
      item.put("toolCall", event.toolCall() == null ? null : toolCallToTree(event.toolCall()));
      events.add(item);
    }
    tree.put("streamEvents", events);

    List<Map<String, Object>> chunks = new ArrayList<>(response.streamChunks().size());
    for (CassetteRecord.ChatPayload chunk : response.streamChunks()) {
      chunks.add(chatPayloadToTree(chunk));
    }
    tree.put("streamChunks", chunks);
    tree.put("metadata", chatMetadataToTree(response.metadata()));
    return tree;
  }

  private static Map<String, Object> aiMessageToTree(CassetteRecord.AiMessagePayload aiMessage) {
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("text", aiMessage.text());
    tree.put("thinking", aiMessage.thinking());
    List<Map<String, Object>> tools = new ArrayList<>(aiMessage.toolExecutionRequests().size());
    for (CassetteRecord.ToolCall tool : aiMessage.toolExecutionRequests()) {
      tools.add(toolCallToTree(tool));
    }
    tree.put("toolExecutionRequests", tools);
    tree.put("attributes", new LinkedHashMap<>(aiMessage.attributes()));
    return tree;
  }

  private static Map<String, Object> toolCallToTree(CassetteRecord.ToolCall tool) {
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("id", tool.id());
    tree.put("type", tool.type());
    tree.put("name", tool.name());
    tree.put("arguments", tool.arguments());
    return tree;
  }

  private static Map<String, Object> generationMetadataToTree(
      CassetteRecord.GenerationMetadata metadata) {
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("finishReason", metadata.finishReason());
    tree.put("contentFilters", new ArrayList<>(metadata.contentFilters()));
    tree.put("attributes", new LinkedHashMap<>(metadata.attributes()));
    return tree;
  }

  private static Map<String, Object> chatMetadataToTree(CassetteRecord.ChatMetadata metadata) {
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("id", metadata.id());
    tree.put("modelName", metadata.modelName());
    tree.put("finishReason", metadata.finishReason());
    tree.put("tokenUsage", tokenUsageToTree(metadata.tokenUsage()));
    tree.put("attributes", new LinkedHashMap<>(metadata.attributes()));
    tree.put("rateLimit", rateLimitToTree(metadata.rateLimit()));
    List<Map<String, Object>> promptMetadata = new ArrayList<>(metadata.promptMetadata().size());
    for (CassetteRecord.PromptFilter filter : metadata.promptMetadata()) {
      Map<String, Object> item = new LinkedHashMap<>();
      item.put("promptIndex", filter.promptIndex());
      item.put("contentFilterMetadata", filter.contentFilterMetadata());
      promptMetadata.add(item);
    }
    tree.put("promptMetadata", promptMetadata);
    return tree;
  }

  private static Map<String, Object> tokenUsageToTree(CassetteRecord.TokenUsage tokenUsage) {
    if (tokenUsage == null) {
      return null;
    }
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("inputTokenCount", tokenUsage.inputTokenCount());
    tree.put("outputTokenCount", tokenUsage.outputTokenCount());
    tree.put("totalTokenCount", tokenUsage.totalTokenCount());
    tree.put("nativeUsage", tokenUsage.nativeUsage());
    return tree;
  }

  private static Map<String, Object> rateLimitToTree(CassetteRecord.RateLimit rateLimit) {
    if (rateLimit == null) {
      return null;
    }
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("requestsLimit", rateLimit.requestsLimit());
    tree.put("requestsRemaining", rateLimit.requestsRemaining());
    tree.put("requestsResetMillis", rateLimit.requestsResetMillis());
    tree.put("tokensLimit", rateLimit.tokensLimit());
    tree.put("tokensRemaining", rateLimit.tokensRemaining());
    tree.put("tokensResetMillis", rateLimit.tokensResetMillis());
    return tree;
  }

  private static float[] toFloatArray(List<?> values) {
    float[] result = new float[values.size()];
    for (int i = 0; i < values.size(); i++) {
      result[i] = requireNumber(values.get(i), "vector[" + i + "]").floatValue();
    }
    return result;
  }

  private static CassetteRecord.ChatPayload toChatPayload(Object raw) {
    Map<?, ?> tree = requireMap(raw, "response");
    return new CassetteRecord.ChatPayload(
        toAiMessage(tree.get("aiMessage")),
        toChatMetadata(tree.get("metadata")),
        toGenerationMetadata(tree.get("generationMetadata")),
        toAdditionalGenerations(tree.get("additionalGenerations")),
        toStreamEvents(tree.get("streamEvents")),
        toStreamChunks(tree.get("streamChunks")));
  }

  private static CassetteRecord.AiMessagePayload toAiMessage(Object raw) {
    Map<?, ?> tree = requireMap(raw, "response.aiMessage");
    List<CassetteRecord.ToolCall> tools = new ArrayList<>();
    Object rawTools = tree.get("toolExecutionRequests");
    if (rawTools instanceof List<?> list) {
      for (Object rawTool : list) {
        tools.add(toRequiredToolCall(rawTool, "toolExecutionRequests[]"));
      }
    }
    return new CassetteRecord.AiMessagePayload(
        asString(tree.get("text")),
        asString(tree.get("thinking")),
        tools,
        toObjectMap(tree.get("attributes")));
  }

  private static CassetteRecord.ChatMetadata toChatMetadata(Object raw) {
    if (raw == null) {
      return CassetteRecord.ChatMetadata.empty();
    }
    Map<?, ?> tree = requireMap(raw, "response.metadata");
    return new CassetteRecord.ChatMetadata(
        asString(tree.get("id")),
        asString(tree.get("modelName")),
        toTokenUsage(tree.get("tokenUsage")),
        asString(tree.get("finishReason")),
        toObjectMap(tree.get("attributes")),
        toRateLimit(tree.get("rateLimit")),
        toPromptMetadata(tree.get("promptMetadata")));
  }

  private static CassetteRecord.TokenUsage toTokenUsage(Object raw) {
    if (raw == null) {
      return null;
    }
    Map<?, ?> tree = requireMap(raw, "response.metadata.tokenUsage");
    return new CassetteRecord.TokenUsage(
        asInteger(tree.get("inputTokenCount")),
        asInteger(tree.get("outputTokenCount")),
        asInteger(tree.get("totalTokenCount")),
        tree.get("nativeUsage"));
  }

  private static CassetteRecord.GenerationMetadata toGenerationMetadata(Object raw) {
    if (raw == null) {
      return CassetteRecord.GenerationMetadata.empty();
    }
    Map<?, ?> tree = requireMap(raw, "generationMetadata");
    LinkedHashSet<String> filters = new LinkedHashSet<>();
    if (tree.get("contentFilters") instanceof List<?> list) {
      for (Object item : list) {
        filters.add(String.valueOf(item));
      }
    }
    return new CassetteRecord.GenerationMetadata(
        asString(tree.get("finishReason")), filters, toObjectMap(tree.get("attributes")));
  }

  private static List<CassetteRecord.ChatGenerationPayload> toAdditionalGenerations(Object raw) {
    if (!(raw instanceof List<?> list)) {
      return List.of();
    }
    List<CassetteRecord.ChatGenerationPayload> result = new ArrayList<>();
    for (Object item : list) {
      Map<?, ?> tree = requireMap(item, "additionalGenerations[]");
      result.add(
          new CassetteRecord.ChatGenerationPayload(
              toAiMessage(tree.get("aiMessage")), toGenerationMetadata(tree.get("metadata"))));
    }
    return result;
  }

  private static List<CassetteRecord.StreamEvent> toStreamEvents(Object raw) {
    if (!(raw instanceof List<?> list)) {
      return List.of();
    }
    List<CassetteRecord.StreamEvent> result = new ArrayList<>();
    for (Object item : list) {
      Map<?, ?> tree = requireMap(item, "streamEvents[]");
      result.add(
          new CassetteRecord.StreamEvent(
              requireString(tree.get("type"), "streamEvents[].type"),
              asString(tree.get("text")),
              asInteger(tree.get("index")),
              toToolCall(tree.get("toolCall"))));
    }
    return result;
  }

  private static List<CassetteRecord.ChatPayload> toStreamChunks(Object raw) {
    if (!(raw instanceof List<?> list)) {
      return List.of();
    }
    List<CassetteRecord.ChatPayload> result = new ArrayList<>();
    for (Object item : list) {
      result.add(toChatPayload(item));
    }
    return result;
  }

  private static CassetteRecord.ToolCall toToolCall(Object raw) {
    return raw == null ? null : toRequiredToolCall(raw, "toolCall");
  }

  private static CassetteRecord.ToolCall toRequiredToolCall(Object raw, String field) {
    Map<?, ?> tree = requireMap(raw, field);
    return new CassetteRecord.ToolCall(
        asString(tree.get("id")),
        asString(tree.get("type")),
        requireString(tree.get("name"), field + ".name"),
        asString(tree.get("arguments")));
  }

  private static CassetteRecord.RateLimit toRateLimit(Object raw) {
    if (raw == null) {
      return null;
    }
    Map<?, ?> tree = requireMap(raw, "rateLimit");
    return new CassetteRecord.RateLimit(
        asLong(tree.get("requestsLimit")),
        asLong(tree.get("requestsRemaining")),
        asLong(tree.get("requestsResetMillis")),
        asLong(tree.get("tokensLimit")),
        asLong(tree.get("tokensRemaining")),
        asLong(tree.get("tokensResetMillis")));
  }

  private static List<CassetteRecord.PromptFilter> toPromptMetadata(Object raw) {
    if (!(raw instanceof List<?> list)) {
      return List.of();
    }
    List<CassetteRecord.PromptFilter> result = new ArrayList<>();
    for (Object item : list) {
      Map<?, ?> tree = requireMap(item, "promptMetadata[]");
      Integer index = asInteger(tree.get("promptIndex"));
      result.add(
          new CassetteRecord.PromptFilter(
              index == null ? 0 : index, tree.get("contentFilterMetadata")));
    }
    return result;
  }

  private static Map<String, Object> toObjectMap(Object raw) {
    if (raw == null) {
      return Map.of();
    }
    Map<?, ?> tree = requireMap(raw, "attributes");
    Map<String, Object> result = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : tree.entrySet()) {
      result.put(String.valueOf(entry.getKey()), entry.getValue());
    }
    return result;
  }

  private static Map<?, ?> requireMap(Object raw, String field) {
    if (raw instanceof Map<?, ?> map) {
      return map;
    }
    throw new IllegalArgumentException("expected object field: " + field);
  }

  private static List<?> requireList(Object raw, String field) {
    if (raw instanceof List<?> list) {
      return list;
    }
    throw new IllegalArgumentException("expected array field: " + field);
  }

  private static String requireString(Object raw, String field) {
    if (raw instanceof String value) {
      return value;
    }
    throw new IllegalArgumentException("expected string field: " + field);
  }

  private static Number requireNumber(Object raw, String field) {
    if (raw instanceof Number value) {
      return value;
    }
    throw new IllegalArgumentException("expected number field: " + field);
  }

  private static String asString(Object raw) {
    return raw instanceof String value ? value : null;
  }

  private static Integer asInteger(Object raw) {
    return raw instanceof Number value ? value.intValue() : null;
  }

  private static Long asLong(Object raw) {
    return raw instanceof Number value ? value.longValue() : null;
  }
}
