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
import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRReplayPolicy;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.PartialToolCallContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

/** LangChain4j streaming chat wrapper that records and replays callback sequences. */
public final class VCRStreamingChatModel implements StreamingChatModel {

  private static final String TYPE_CHAT_STREAM = "chat_stream";

  private final StreamingChatModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger callCounter = new AtomicInteger();

  /** Creates a VCR wrapper around a real LangChain4j streaming chat model. */
  public VCRStreamingChatModel(
      StreamingChatModel delegate,
      String testId,
      VCRMode mode,
      String modelName,
      CassetteStore store) {
    this.delegate = delegate;
    this.testId = testId;
    this.mode = mode;
    this.modelName = modelName;
    this.store = store;
  }

  @Override
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    if (mode == VCRMode.OFF) {
      delegate.doChat(request, handler);
      return;
    }
    CassetteKey key = new CassetteKey(TYPE_CHAT_STREAM, testId, callCounter.incrementAndGet());
    String signature =
        LangChain4jRequestSignatures.chat(
            "chat_stream", modelName, request, delegate.defaultRequestParameters());
    Optional<CassetteRecord> existing = store.retrieve(key);
    validateType(existing, key);
    if (VCRReplayPolicy.shouldReplay(mode, existing, key, signature)) {
      replay((CassetteRecord.Chat) existing.orElseThrow(), handler);
      return;
    }
    delegate.doChat(request, new RecordingHandler(key, request, signature, handler));
  }

  @Override
  public ChatRequestParameters defaultRequestParameters() {
    return delegate.defaultRequestParameters();
  }

  private void replay(CassetteRecord.Chat chat, StreamingChatResponseHandler handler) {
    for (CassetteRecord.StreamEvent event : chat.response().streamEvents()) {
      switch (event.type()) {
        case "partial_response" -> handler.onPartialResponse(event.text());
        case "partial_thinking" -> handler.onPartialThinking(new PartialThinking(event.text()));
        case "partial_tool_call" -> handler.onPartialToolCall(toPartialToolCall(event));
        case "complete_tool_call" -> handler.onCompleteToolCall(toCompleteToolCall(event));
        default ->
            throw new IllegalStateException("Unknown streaming cassette event: " + event.type());
      }
    }
    handler.onCompleteResponse(VCRChatModel.toChatResponse(chat.response()));
  }

  private static PartialToolCall toPartialToolCall(CassetteRecord.StreamEvent event) {
    CassetteRecord.ToolCall tool = event.toolCall();
    return PartialToolCall.builder()
        .index(event.index())
        .id(tool.id())
        .name(tool.name())
        .partialArguments(tool.arguments())
        .build();
  }

  private static CompleteToolCall toCompleteToolCall(CassetteRecord.StreamEvent event) {
    CassetteRecord.ToolCall tool = event.toolCall();
    ToolExecutionRequest request =
        ToolExecutionRequest.builder()
            .id(tool.id())
            .name(tool.name())
            .arguments(tool.arguments())
            .build();
    return new CompleteToolCall(event.index(), request);
  }

  private static void validateType(Optional<CassetteRecord> existing, CassetteKey key) {
    if (existing.isPresent() && !(existing.get() instanceof CassetteRecord.Chat)) {
      throw new IllegalStateException(
          "Expected Chat cassette for key "
              + key.serializedKey()
              + " but got "
              + existing.get().getClass().getSimpleName());
    }
  }

  private final class RecordingHandler implements StreamingChatResponseHandler {
    private final CassetteKey key;
    private final ChatRequest request;
    private final String signature;
    private final StreamingChatResponseHandler downstream;
    private final List<CassetteRecord.StreamEvent> events = new ArrayList<>();

    private RecordingHandler(
        CassetteKey key,
        ChatRequest request,
        String signature,
        StreamingChatResponseHandler downstream) {
      this.key = key;
      this.request = request;
      this.signature = signature;
      this.downstream = downstream;
    }

    @Override
    public void onPartialResponse(String text) {
      events.add(new CassetteRecord.StreamEvent("partial_response", text, null, null));
      downstream.onPartialResponse(text);
    }

    @Override
    public void onPartialResponse(PartialResponse response, PartialResponseContext context) {
      events.add(new CassetteRecord.StreamEvent("partial_response", response.text(), null, null));
      downstream.onPartialResponse(response, context);
    }

    @Override
    public void onPartialThinking(PartialThinking thinking) {
      events.add(new CassetteRecord.StreamEvent("partial_thinking", thinking.text(), null, null));
      downstream.onPartialThinking(thinking);
    }

    @Override
    public void onPartialThinking(PartialThinking thinking, PartialThinkingContext context) {
      events.add(new CassetteRecord.StreamEvent("partial_thinking", thinking.text(), null, null));
      downstream.onPartialThinking(thinking, context);
    }

    @Override
    public void onPartialToolCall(PartialToolCall toolCall) {
      events.add(partialToolEvent(toolCall));
      downstream.onPartialToolCall(toolCall);
    }

    @Override
    public void onPartialToolCall(PartialToolCall toolCall, PartialToolCallContext context) {
      events.add(partialToolEvent(toolCall));
      downstream.onPartialToolCall(toolCall, context);
    }

    @Override
    public void onCompleteToolCall(CompleteToolCall toolCall) {
      ToolExecutionRequest tool = toolCall.toolExecutionRequest();
      events.add(
          new CassetteRecord.StreamEvent(
              "complete_tool_call",
              null,
              toolCall.index(),
              new CassetteRecord.ToolCall(tool.id(), tool.name(), tool.arguments())));
      downstream.onCompleteToolCall(toolCall);
    }

    @Override
    public void onCompleteResponse(ChatResponse response) {
      CassetteRecord.ChatPayload complete = VCRChatModel.toPayload(response);
      CassetteRecord.ChatPayload recorded =
          new CassetteRecord.ChatPayload(
              complete.aiMessage(),
              complete.metadata(),
              complete.generationMetadata(),
              complete.additionalGenerations(),
              events,
              List.of());
      store.store(
          key,
          new CassetteRecord.Chat(
              testId,
              modelName,
              System.currentTimeMillis(),
              String.valueOf(request.messages()),
              recorded,
              signature));
      downstream.onCompleteResponse(response);
    }

    @Override
    public void onError(Throwable error) {
      downstream.onError(error);
    }

    private CassetteRecord.StreamEvent partialToolEvent(PartialToolCall toolCall) {
      return new CassetteRecord.StreamEvent(
          "partial_tool_call",
          null,
          toolCall.index(),
          new CassetteRecord.ToolCall(toolCall.id(), toolCall.name(), toolCall.partialArguments()));
    }
  }
}
