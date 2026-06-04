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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class VCRChatModelTest {

  @Mock ChatModel delegate;

  CassetteStore store;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
  }

  @Test
  void recordsAndReplaysChatResponse() {
    ToolExecutionRequest tool =
        ToolExecutionRequest.builder()
            .id("call-1")
            .name("lookup")
            .arguments("{\"query\":\"meaning\"}")
            .build();
    when(delegate.doChat(any(ChatRequest.class)))
        .thenReturn(
            ChatResponse.builder()
                .aiMessage(
                    AiMessage.builder()
                        .text("42")
                        .thinking("hidden")
                        .toolExecutionRequests(List.of(tool))
                        .attributes(Map.of("provider", "fake"))
                        .build())
                .id("resp-1")
                .modelName("test-model")
                .tokenUsage(new TokenUsage(3, 4, 7))
                .finishReason(FinishReason.TOOL_EXECUTION)
                .build());

    ChatRequest request = ChatRequest.builder().messages(UserMessage.from("meaning?")).build();

    VCRChatModel recorder = new VCRChatModel(delegate, "T:c", VCRMode.RECORD, "m", store);
    ChatResponse recorded = recorder.doChat(request);
    assertThat(recorded.aiMessage().text()).isEqualTo("42");

    VCRChatModel player = new VCRChatModel(delegate, "T:c", VCRMode.PLAYBACK, "m", store);
    ChatResponse played = player.doChat(request);
    assertThat(played.aiMessage().text()).isEqualTo("42");
    assertThat(played.aiMessage().thinking()).isEqualTo("hidden");
    assertThat(played.aiMessage().toolExecutionRequests()).containsExactly(tool);
    assertThat(played.aiMessage().attributes()).containsEntry("provider", "fake");
    assertThat(played.id()).isEqualTo("resp-1");
    assertThat(played.modelName()).isEqualTo("test-model");
    assertThat(played.tokenUsage()).isEqualTo(new TokenUsage(3, 4, 7));
    assertThat(played.finishReason()).isEqualTo(FinishReason.TOOL_EXECUTION);
    verify(delegate, times(1)).doChat(any(ChatRequest.class));
  }

  @Test
  void playbackThrowsWhenMissing() {
    ChatRequest request = ChatRequest.builder().messages(UserMessage.from("x")).build();

    VCRChatModel player = new VCRChatModel(delegate, "T:miss", VCRMode.PLAYBACK, "m", store);
    assertThatThrownBy(() -> player.doChat(request))
        .isInstanceOf(VCRCassetteMissingException.class);
    verify(delegate, never()).doChat(any(ChatRequest.class));
  }
}
