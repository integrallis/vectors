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

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class VCRStreamingChatModelTest {

  @Mock StreamingChatModel delegate;

  CassetteStore store;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
  }

  @Test
  void recordsAndReplaysStreamingCallbacksAndFinalResponse() {
    ChatResponse complete =
        ChatResponse.builder()
            .aiMessage(AiMessage.builder().text("hello").thinking("why").build())
            .build();
    doAnswer(
            invocation -> {
              StreamingChatResponseHandler handler = invocation.getArgument(1);
              handler.onPartialThinking(new PartialThinking("why"));
              handler.onPartialResponse("hel");
              handler.onPartialResponse("lo");
              handler.onPartialToolCall(
                  PartialToolCall.builder()
                      .index(0)
                      .id("call-1")
                      .name("lookup")
                      .partialArguments("{\"id\":")
                      .build());
              handler.onCompleteToolCall(
                  new CompleteToolCall(
                      0,
                      ToolExecutionRequest.builder()
                          .id("call-1")
                          .name("lookup")
                          .arguments("{\"id\":42}")
                          .build()));
              handler.onCompleteResponse(complete);
              return null;
            })
        .when(delegate)
        .doChat(any(ChatRequest.class), any(StreamingChatResponseHandler.class));
    ChatRequest request =
        ChatRequest.builder().messages(UserMessage.from("say hello")).temperature(0.0).build();

    CapturingHandler recorded = new CapturingHandler();
    new VCRStreamingChatModel(delegate, "T:stream", VCRMode.RECORD, "gpt", store)
        .doChat(request, recorded);
    CapturingHandler replayed = new CapturingHandler();
    new VCRStreamingChatModel(delegate, "T:stream", VCRMode.PLAYBACK, "gpt", store)
        .doChat(request, replayed);

    assertThat(recorded.events)
        .containsExactly(
            "thinking:why",
            "text:hel",
            "text:lo",
            "partial-tool:0:lookup:{\"id\":",
            "complete-tool:0:lookup:{\"id\":42}");
    assertThat(replayed.events).isEqualTo(recorded.events);
    assertThat(replayed.complete.aiMessage().text()).isEqualTo("hello");
    assertThat(replayed.complete.aiMessage().thinking()).isEqualTo("why");
    verify(delegate, times(1))
        .doChat(any(ChatRequest.class), any(StreamingChatResponseHandler.class));
  }

  private static final class CapturingHandler implements StreamingChatResponseHandler {
    private final List<String> events = new ArrayList<>();
    private ChatResponse complete;

    @Override
    public void onPartialResponse(String partialResponse) {
      events.add("text:" + partialResponse);
    }

    @Override
    public void onPartialThinking(PartialThinking partialThinking) {
      events.add("thinking:" + partialThinking.text());
    }

    @Override
    public void onPartialToolCall(PartialToolCall partialToolCall) {
      events.add(
          "partial-tool:"
              + partialToolCall.index()
              + ':'
              + partialToolCall.name()
              + ':'
              + partialToolCall.partialArguments());
    }

    @Override
    public void onCompleteToolCall(CompleteToolCall completeToolCall) {
      events.add(
          "complete-tool:"
              + completeToolCall.index()
              + ':'
              + completeToolCall.toolExecutionRequest().name()
              + ':'
              + completeToolCall.toolExecutionRequest().arguments());
    }

    @Override
    public void onCompleteResponse(ChatResponse completeResponse) {
      complete = completeResponse;
    }

    @Override
    public void onError(Throwable error) {
      throw new AssertionError(error);
    }
  }
}
