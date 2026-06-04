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

import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * LangChain4j {@link ChatModel} wrapper that records/replays chat responses through a {@link
 * CassetteStore}.
 */
public final class VCRChatModel implements ChatModel {

  private static final String TYPE_CHAT = "chat";

  private final ChatModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger callCounter = new AtomicInteger();

  /**
   * @param delegate underlying LangChain4j chat model
   * @param testId test identifier
   * @param mode VCR mode
   * @param modelName model name for cassette metadata
   * @param store cassette store
   */
  public VCRChatModel(
      ChatModel delegate, String testId, VCRMode mode, String modelName, CassetteStore store) {
    this.delegate = delegate;
    this.testId = testId;
    this.mode = mode;
    this.modelName = modelName;
    this.store = store;
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    if (mode == VCRMode.OFF) {
      return delegate.doChat(request);
    }
    CassetteKey key = new CassetteKey(TYPE_CHAT, testId, callCounter.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> hit = store.retrieve(key);
      if (hit.isPresent()) {
        if (!(hit.get() instanceof CassetteRecord.Chat c)) {
          throw new IllegalStateException(
              "Expected Chat cassette for key "
                  + key.serializedKey()
                  + " but got "
                  + hit.get().getClass().getSimpleName());
        }
        return toChatResponse(c.response());
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    ChatResponse response = delegate.doChat(request);
    String prompt = String.valueOf(request.messages());
    store.store(
        key,
        new CassetteRecord.Chat(
            testId, modelName, System.currentTimeMillis(), prompt, toPayload(response)));
    return response;
  }

  private static CassetteRecord.ChatPayload toPayload(ChatResponse response) {
    AiMessage ai = response.aiMessage();
    CassetteRecord.AiMessagePayload aiPayload =
        new CassetteRecord.AiMessagePayload(
            ai == null ? null : ai.text(),
            ai == null ? null : ai.thinking(),
            toToolCalls(ai),
            ai == null ? null : ai.attributes());
    TokenUsage usage = response.tokenUsage();
    CassetteRecord.TokenUsage tokenUsage =
        usage == null
            ? null
            : new CassetteRecord.TokenUsage(
                usage.inputTokenCount(), usage.outputTokenCount(), usage.totalTokenCount());
    FinishReason finishReason = response.finishReason();
    CassetteRecord.ChatMetadata metadata =
        new CassetteRecord.ChatMetadata(
            response.id(),
            response.modelName(),
            tokenUsage,
            finishReason == null ? null : finishReason.name());
    return new CassetteRecord.ChatPayload(aiPayload, metadata);
  }

  private static List<CassetteRecord.ToolCall> toToolCalls(AiMessage ai) {
    if (ai == null || ai.toolExecutionRequests() == null) {
      return List.of();
    }
    return ai.toolExecutionRequests().stream()
        .map(t -> new CassetteRecord.ToolCall(t.id(), t.name(), t.arguments()))
        .toList();
  }

  private static ChatResponse toChatResponse(CassetteRecord.ChatPayload payload) {
    CassetteRecord.AiMessagePayload ai = payload.aiMessage();
    AiMessage aiMessage =
        AiMessage.builder()
            .text(ai.text())
            .thinking(ai.thinking())
            .toolExecutionRequests(toToolExecutionRequests(ai.toolExecutionRequests()))
            .attributes(ai.attributes())
            .build();
    CassetteRecord.ChatMetadata metadata = payload.metadata();
    ChatResponse.Builder builder = ChatResponse.builder().aiMessage(aiMessage);
    if (metadata.id() != null) {
      builder.id(metadata.id());
    }
    if (metadata.modelName() != null) {
      builder.modelName(metadata.modelName());
    }
    if (metadata.tokenUsage() != null) {
      builder.tokenUsage(
          new TokenUsage(
              metadata.tokenUsage().inputTokenCount(),
              metadata.tokenUsage().outputTokenCount(),
              metadata.tokenUsage().totalTokenCount()));
    }
    if (metadata.finishReason() != null) {
      builder.finishReason(FinishReason.valueOf(metadata.finishReason()));
    }
    return builder.build();
  }

  private static List<ToolExecutionRequest> toToolExecutionRequests(
      List<CassetteRecord.ToolCall> toolCalls) {
    return toolCalls.stream()
        .map(
            t ->
                ToolExecutionRequest.builder()
                    .id(t.id())
                    .name(t.name())
                    .arguments(t.arguments())
                    .build())
        .toList();
  }

  /**
   * @return the underlying delegate
   */
  public ChatModel getDelegate() {
    return delegate;
  }
}
