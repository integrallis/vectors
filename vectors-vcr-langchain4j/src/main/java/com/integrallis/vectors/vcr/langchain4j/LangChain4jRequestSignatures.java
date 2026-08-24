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
package com.integrallis.vectors.vcr.langchain4j;

import com.integrallis.vectors.vcr.RequestSignature;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class LangChain4jRequestSignatures {

  private LangChain4jRequestSignatures() {}

  static String chat(
      String modelLabel, ChatRequest request, ChatRequestParameters defaultParameters) {
    return chat("chat", modelLabel, request, defaultParameters);
  }

  static String chat(
      String operation,
      String modelLabel,
      ChatRequest request,
      ChatRequestParameters defaultParameters) {
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("operation", operation);
    tree.put("modelLabel", modelLabel);
    tree.put(
        "messages",
        request.messages().stream().map(LangChain4jRequestSignatures::message).toList());
    tree.put("requestParameters", parameters(request.parameters()));
    tree.put("defaultParameters", parameters(defaultParameters));
    return RequestSignature.create(tree);
  }

  static String embedding(String modelLabel, String text) {
    return RequestSignature.create(
        Map.of("operation", "embedding", "modelLabel", modelLabel, "text", text));
  }

  static String batchEmbedding(String modelLabel, List<String> texts) {
    return RequestSignature.create(
        Map.of("operation", "batch_embedding", "modelLabel", modelLabel, "texts", texts));
  }

  private static Map<String, Object> message(ChatMessage message) {
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("type", message.type().name());
    tree.put("value", String.valueOf(message));
    return tree;
  }

  private static Map<String, Object> parameters(ChatRequestParameters parameters) {
    if (parameters == null) {
      return Map.of();
    }
    Map<String, Object> tree = new LinkedHashMap<>();
    tree.put("modelName", parameters.modelName());
    tree.put("temperature", parameters.temperature());
    tree.put("topP", parameters.topP());
    tree.put("topK", parameters.topK());
    tree.put("frequencyPenalty", parameters.frequencyPenalty());
    tree.put("presencePenalty", parameters.presencePenalty());
    tree.put("maxOutputTokens", parameters.maxOutputTokens());
    tree.put("stopSequences", strings(parameters.stopSequences()));
    tree.put("toolSpecifications", strings(parameters.toolSpecifications()));
    tree.put("toolChoice", parameters.toolChoice() == null ? null : parameters.toolChoice().name());
    tree.put("responseFormat", String.valueOf(parameters.responseFormat()));
    return tree;
  }

  private static List<String> strings(List<?> values) {
    return values == null ? List.of() : values.stream().map(String::valueOf).toList();
  }
}
