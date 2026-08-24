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

import com.integrallis.vectors.vcr.CassetteRecord;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.PromptMetadata;
import org.springframework.ai.chat.metadata.RateLimit;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;

/** Converts complete Spring AI responses to and from framework-neutral cassette payloads. */
final class SpringAIChatPayloadMapper {

  private SpringAIChatPayloadMapper() {}

  static CassetteRecord.ChatPayload toPayload(ChatResponse response) {
    List<Generation> generations = response.getResults();
    Generation primary = generations.getFirst();
    List<CassetteRecord.ChatGenerationPayload> additional = new ArrayList<>();
    generations.stream()
        .skip(1)
        .map(
            generation ->
                new CassetteRecord.ChatGenerationPayload(
                    toMessage(generation.getOutput()),
                    toGenerationMetadata(generation.getMetadata())))
        .forEach(additional::add);
    return new CassetteRecord.ChatPayload(
        toMessage(primary.getOutput()),
        toResponseMetadata(response.getMetadata()),
        toGenerationMetadata(primary.getMetadata()),
        additional,
        List.of(),
        List.of());
  }

  static ChatResponse toResponse(CassetteRecord.ChatPayload payload) {
    List<Generation> generations = new ArrayList<>();
    generations.add(
        new Generation(
            toMessage(payload.aiMessage()), toGenerationMetadata(payload.generationMetadata())));
    payload.additionalGenerations().stream()
        .map(
            generation ->
                new Generation(
                    toMessage(generation.aiMessage()), toGenerationMetadata(generation.metadata())))
        .forEach(generations::add);
    return new ChatResponse(generations, toResponseMetadata(payload.metadata()));
  }

  private static CassetteRecord.AiMessagePayload toMessage(AssistantMessage message) {
    List<CassetteRecord.ToolCall> toolCalls =
        message.getToolCalls().stream()
            .map(
                tool ->
                    new CassetteRecord.ToolCall(
                        tool.id(), tool.type(), tool.name(), tool.arguments()))
            .toList();
    return new CassetteRecord.AiMessagePayload(
        message.getText(), null, toolCalls, message.getMetadata());
  }

  private static AssistantMessage toMessage(CassetteRecord.AiMessagePayload payload) {
    List<AssistantMessage.ToolCall> toolCalls =
        payload.toolExecutionRequests().stream()
            .map(
                tool ->
                    new AssistantMessage.ToolCall(
                        tool.id(), tool.type(), tool.name(), tool.arguments()))
            .toList();
    return AssistantMessage.builder()
        .content(payload.text())
        .properties(payload.attributes())
        .toolCalls(toolCalls)
        .build();
  }

  private static CassetteRecord.GenerationMetadata toGenerationMetadata(
      ChatGenerationMetadata metadata) {
    if (metadata == null) {
      return CassetteRecord.GenerationMetadata.empty();
    }
    return new CassetteRecord.GenerationMetadata(
        metadata.getFinishReason(), metadata.getContentFilters(), entries(metadata.entrySet()));
  }

  private static ChatGenerationMetadata toGenerationMetadata(
      CassetteRecord.GenerationMetadata metadata) {
    return ChatGenerationMetadata.builder()
        .finishReason(metadata.finishReason())
        .contentFilters(metadata.contentFilters())
        .metadata(metadata.attributes())
        .build();
  }

  private static CassetteRecord.ChatMetadata toResponseMetadata(ChatResponseMetadata metadata) {
    if (metadata == null) {
      return CassetteRecord.ChatMetadata.empty();
    }
    Usage usage = metadata.getUsage();
    CassetteRecord.TokenUsage tokenUsage =
        usage == null
            ? null
            : new CassetteRecord.TokenUsage(
                usage.getPromptTokens(),
                usage.getCompletionTokens(),
                usage.getTotalTokens(),
                usage.getNativeUsage());
    RateLimit rateLimit = metadata.getRateLimit();
    CassetteRecord.RateLimit recordedRateLimit =
        rateLimit == null
            ? null
            : new CassetteRecord.RateLimit(
                rateLimit.getRequestsLimit(),
                rateLimit.getRequestsRemaining(),
                millis(rateLimit.getRequestsReset()),
                rateLimit.getTokensLimit(),
                rateLimit.getTokensRemaining(),
                millis(rateLimit.getTokensReset()));
    List<CassetteRecord.PromptFilter> promptMetadata = new ArrayList<>();
    if (metadata.getPromptMetadata() != null) {
      metadata
          .getPromptMetadata()
          .forEach(
              filter ->
                  promptMetadata.add(
                      new CassetteRecord.PromptFilter(
                          filter.getPromptIndex(), filter.getContentFilterMetadata())));
    }
    return new CassetteRecord.ChatMetadata(
        metadata.getId(),
        metadata.getModel(),
        tokenUsage,
        null,
        entries(metadata.entrySet()),
        recordedRateLimit,
        promptMetadata);
  }

  private static ChatResponseMetadata toResponseMetadata(CassetteRecord.ChatMetadata metadata) {
    ChatResponseMetadata.Builder builder =
        ChatResponseMetadata.builder()
            .id(metadata.id())
            .model(metadata.modelName())
            .metadata(metadata.attributes());
    if (metadata.tokenUsage() != null) {
      CassetteRecord.TokenUsage usage = metadata.tokenUsage();
      builder.usage(
          new DefaultUsage(
              usage.inputTokenCount(),
              usage.outputTokenCount(),
              usage.totalTokenCount(),
              usage.nativeUsage()));
    }
    if (metadata.rateLimit() != null) {
      builder.rateLimit(new RecordedRateLimit(metadata.rateLimit()));
    }
    if (!metadata.promptMetadata().isEmpty()) {
      builder.promptMetadata(
          PromptMetadata.of(
              metadata.promptMetadata().stream()
                  .map(
                      filter ->
                          PromptMetadata.PromptFilterMetadata.from(
                              filter.promptIndex(), filter.contentFilterMetadata()))
                  .toList()));
    }
    return builder.build();
  }

  private static Map<String, Object> entries(java.util.Set<Map.Entry<String, Object>> entries) {
    Map<String, Object> result = new LinkedHashMap<>();
    entries.forEach(entry -> result.put(entry.getKey(), entry.getValue()));
    return result;
  }

  private static Long millis(Duration duration) {
    return duration == null ? null : duration.toMillis();
  }

  private record RecordedRateLimit(CassetteRecord.RateLimit value) implements RateLimit {
    @Override
    public Long getRequestsLimit() {
      return value.requestsLimit();
    }

    @Override
    public Long getRequestsRemaining() {
      return value.requestsRemaining();
    }

    @Override
    public Duration getRequestsReset() {
      return duration(value.requestsResetMillis());
    }

    @Override
    public Long getTokensLimit() {
      return value.tokensLimit();
    }

    @Override
    public Long getTokensRemaining() {
      return value.tokensRemaining();
    }

    @Override
    public Duration getTokensReset() {
      return duration(value.tokensResetMillis());
    }

    private static Duration duration(Long millis) {
      return millis == null ? null : Duration.ofMillis(millis);
    }
  }
}
