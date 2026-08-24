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
package com.integrallis.vectors.vcr.springai;

import com.integrallis.vectors.vcr.RequestSignature;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.model.tool.StructuredOutputChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;

/** Builds stable signatures from Spring AI requests and effective model options. */
final class SpringAIRequestSignatures {

  private SpringAIRequestSignatures() {}

  static String chat(String operation, String modelName, Prompt prompt, ChatOptions defaults) {
    Map<String, Object> request = new LinkedHashMap<>();
    request.put("operation", operation);
    request.put("model", modelName);
    request.put("messages", messages(prompt));
    request.put("options", options(prompt == null ? null : prompt.getOptions()));
    request.put("defaultOptions", options(defaults));
    return RequestSignature.create(request);
  }

  static String embedding(String operation, String modelName, List<String> texts, Object options) {
    Map<String, Object> request = new LinkedHashMap<>();
    request.put("operation", operation);
    request.put("model", modelName);
    request.put("texts", texts);
    if (options instanceof EmbeddingOptions embeddingOptions) {
      request.put(
          "options",
          Map.of(
              "model",
              nullSafe(embeddingOptions.getModel()),
              "dimensions",
              embeddingOptions.getDimensions() == null ? -1 : embeddingOptions.getDimensions()));
    } else {
      request.put("options", stable(options));
    }
    return RequestSignature.create(request);
  }

  private static List<Object> messages(Prompt prompt) {
    if (prompt == null) {
      return List.of();
    }
    List<Object> result = new ArrayList<>();
    for (Message message : prompt.getInstructions()) {
      Map<String, Object> value = new LinkedHashMap<>();
      value.put("type", message.getMessageType().name());
      value.put("text", message.getText());
      value.put("metadata", stable(message.getMetadata()));
      if (message instanceof AssistantMessage assistant) {
        value.put(
            "toolCalls",
            assistant.getToolCalls().stream()
                .map(
                    tool ->
                        Map.<String, Object>of(
                            "id", nullSafe(tool.id()),
                            "type", nullSafe(tool.type()),
                            "name", nullSafe(tool.name()),
                            "arguments", nullSafe(tool.arguments())))
                .toList());
      }
      result.add(value);
    }
    return result;
  }

  private static Object options(ChatOptions options) {
    if (options == null) {
      return null;
    }
    Map<String, Object> value = new LinkedHashMap<>();
    value.put("type", options.getClass().getName());
    value.put("model", options.getModel());
    value.put("frequencyPenalty", options.getFrequencyPenalty());
    value.put("maxTokens", options.getMaxTokens());
    value.put("presencePenalty", options.getPresencePenalty());
    value.put("stopSequences", stable(options.getStopSequences()));
    value.put("temperature", options.getTemperature());
    value.put("topK", options.getTopK());
    value.put("topP", options.getTopP());
    if (options instanceof ToolCallingChatOptions tools) {
      value.put("toolNames", stable(tools.getToolNames()));
      value.put("internalToolExecutionEnabled", tools.getInternalToolExecutionEnabled());
      value.put("toolContext", stable(tools.getToolContext()));
      value.put(
          "toolCallbacks",
          tools.getToolCallbacks() == null
              ? List.of()
              : tools.getToolCallbacks().stream()
                  .map(SpringAIRequestSignatures::toolCallback)
                  .toList());
    }
    if (options instanceof StructuredOutputChatOptions structured) {
      value.put("outputSchema", structured.getOutputSchema());
    }
    return value;
  }

  private static Object stable(Object value) {
    if (value == null
        || value instanceof String
        || value instanceof Number
        || value instanceof Boolean
        || value instanceof Enum<?>) {
      return value;
    }
    if (value instanceof Map<?, ?> map) {
      Map<String, Object> result = new LinkedHashMap<>();
      map.forEach((key, item) -> result.put(String.valueOf(key), stable(item)));
      return result;
    }
    if (value instanceof Iterable<?> iterable) {
      List<Object> result = new ArrayList<>();
      iterable.forEach(item -> result.add(stable(item)));
      return result;
    }
    if (value.getClass().isArray()) {
      List<Object> result = new ArrayList<>();
      for (int i = 0; i < Array.getLength(value); i++) {
        result.add(stable(Array.get(value, i)));
      }
      return result;
    }
    return value.getClass().getName() + ':' + value;
  }

  private static Map<String, Object> toolCallback(ToolCallback callback) {
    Map<String, Object> value = new LinkedHashMap<>();
    value.put("name", callback.getToolDefinition().name());
    value.put("description", callback.getToolDefinition().description());
    value.put("inputSchema", callback.getToolDefinition().inputSchema());
    value.put(
        "returnDirect",
        callback.getToolMetadata() != null && callback.getToolMetadata().returnDirect());
    return value;
  }

  private static String nullSafe(String value) {
    return value == null ? "" : value;
  }
}
