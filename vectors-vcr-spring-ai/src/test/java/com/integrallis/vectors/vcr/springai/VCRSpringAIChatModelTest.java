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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.PromptMetadata;
import org.springframework.ai.chat.metadata.RateLimit;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class VCRSpringAIChatModelTest {

  @Mock ChatModel delegate;

  CassetteStore store;

  @BeforeEach
  void setUp() {
    store = new ExactCassetteStore(new HeapStorageBackend());
  }

  @Test
  void recordAndReplayStringCall() {
    when(delegate.call(anyString())).thenReturn("42");
    VCRSpringAIChatModel recorder =
        new VCRSpringAIChatModel(delegate, "T:c", VCRMode.RECORD, "m", store);
    assertThat(recorder.call("what?")).isEqualTo("42");

    VCRSpringAIChatModel player =
        new VCRSpringAIChatModel(delegate, "T:c", VCRMode.PLAYBACK, "m", store);
    assertThat(player.call("what?")).isEqualTo("42");
    verify(delegate, times(1)).call(anyString());
  }

  @Test
  void playbackOrRecordRefreshesWhenPromptOptionsChange() {
    ChatResponse cold = response("cold");
    ChatResponse hot = response("hot");
    when(delegate.call(any(Prompt.class))).thenReturn(cold, hot);
    Prompt deterministic =
        new Prompt("same prompt", ChatOptions.builder().temperature(0.0).maxTokens(32).build());
    Prompt creative =
        new Prompt("same prompt", ChatOptions.builder().temperature(0.9).maxTokens(32).build());

    new VCRSpringAIChatModel(delegate, "T:signature", VCRMode.RECORD, "gpt", store)
        .call(deterministic);
    ChatResponse unchanged =
        new VCRSpringAIChatModel(delegate, "T:signature", VCRMode.PLAYBACK_OR_RECORD, "gpt", store)
            .call(deterministic);
    ChatResponse changed =
        new VCRSpringAIChatModel(delegate, "T:signature", VCRMode.PLAYBACK_OR_RECORD, "gpt", store)
            .call(creative);

    assertThat(unchanged.getResult().getOutput().getText()).isEqualTo("cold");
    assertThat(changed.getResult().getOutput().getText()).isEqualTo("hot");
    verify(delegate, times(2)).call(any(Prompt.class));
  }

  @Test
  void preservesCompleteBlockingResponse() {
    var toolCall = new AssistantMessage.ToolCall("call-1", "function", "lookup", "{\"id\":42}");
    AssistantMessage message =
        AssistantMessage.builder()
            .content("answer")
            .properties(Map.of("message-id", "msg-1"))
            .toolCalls(List.of(toolCall))
            .build();
    ChatGenerationMetadata generationMetadata =
        ChatGenerationMetadata.builder()
            .finishReason("TOOL_CALLS")
            .contentFilters(Set.of("safe"))
            .metadata("provider-generation", "g-1")
            .build();
    ChatResponseMetadata responseMetadata =
        ChatResponseMetadata.builder()
            .id("response-1")
            .model("gpt-test")
            .usage(new DefaultUsage(7, 5, 12, Map.of("cached_tokens", 2)))
            .rateLimit(rateLimit())
            .promptMetadata(
                PromptMetadata.of(
                    PromptMetadata.PromptFilterMetadata.from(0, Map.of("filter", "safe"))))
            .keyValue("system-fingerprint", "fp-1")
            .build();
    ChatResponse live =
        new ChatResponse(
            List.of(
                new Generation(message, generationMetadata),
                new Generation(new AssistantMessage("alternate"))),
            responseMetadata);
    when(delegate.call(any(Prompt.class))).thenReturn(live);
    Prompt prompt = new Prompt("use the lookup tool");

    new VCRSpringAIChatModel(delegate, "T:full", VCRMode.RECORD, "gpt", store).call(prompt);
    ChatResponse replayed =
        new VCRSpringAIChatModel(delegate, "T:full", VCRMode.PLAYBACK, "gpt", store).call(prompt);

    assertThat(replayed.getResult().getOutput().getToolCalls()).containsExactly(toolCall);
    assertThat(replayed.getResult().getOutput().getMetadata()).containsEntry("message-id", "msg-1");
    assertThat(replayed.getResult().getMetadata().getFinishReason()).isEqualTo("TOOL_CALLS");
    assertThat(replayed.getResult().getMetadata().getContentFilters()).containsExactly("safe");
    Object providerGeneration = replayed.getResult().getMetadata().get("provider-generation");
    assertThat(providerGeneration).isEqualTo("g-1");
    assertThat(replayed.getMetadata().getId()).isEqualTo("response-1");
    assertThat(replayed.getMetadata().getModel()).isEqualTo("gpt-test");
    assertThat(replayed.getMetadata().getUsage().getPromptTokens()).isEqualTo(7);
    assertThat(replayed.getMetadata().getUsage().getCompletionTokens()).isEqualTo(5);
    assertThat(replayed.getMetadata().getRateLimit().getTokensRemaining()).isEqualTo(90L);
    assertThat(replayed.getMetadata().getRateLimit().getTokensReset())
        .isEqualTo(Duration.ofSeconds(2));
    assertThat(
            replayed
                .getMetadata()
                .getPromptMetadata()
                .findByPromptIndex(0)
                .orElseThrow()
                .<Map<String, String>>getContentFilterMetadata())
        .containsEntry("filter", "safe");
    assertThat(replayed.getResults()).hasSize(2);
    assertThat(replayed.getResults().get(1).getOutput().getText()).isEqualTo("alternate");
    Object systemFingerprint = replayed.getMetadata().get("system-fingerprint");
    assertThat(systemFingerprint).isEqualTo("fp-1");
  }

  @Test
  void recordsAndReplaysSpringStreamingChunks() {
    Prompt prompt = new Prompt("stream it");
    when(delegate.stream(any(Prompt.class))).thenReturn(Flux.just(response("hel"), response("lo")));

    List<ChatResponse> recorded =
        new VCRSpringAIChatModel(delegate, "T:stream", VCRMode.RECORD, "gpt", store)
            .stream(prompt).collectList().block();
    List<ChatResponse> replayed =
        new VCRSpringAIChatModel(delegate, "T:stream", VCRMode.PLAYBACK, "gpt", store)
            .stream(prompt).collectList().block();

    assertThat(recorded)
        .extracting(r -> r.getResult().getOutput().getText())
        .containsExactly("hel", "lo");
    assertThat(replayed)
        .extracting(r -> r.getResult().getOutput().getText())
        .containsExactly("hel", "lo");
    verify(delegate, times(1)).stream(any(Prompt.class));
  }

  @Test
  void playbackThrowsWhenMissing() {
    VCRSpringAIChatModel player =
        new VCRSpringAIChatModel(delegate, "T:miss", VCRMode.PLAYBACK, "m", store);
    assertThatThrownBy(() -> player.call("anything"))
        .isInstanceOf(VCRCassetteMissingException.class);
    verify(delegate, never()).call(anyString());
  }

  @Test
  void offModeBypassesStore() {
    when(delegate.call(anyString())).thenReturn("live");
    VCRSpringAIChatModel off = new VCRSpringAIChatModel(delegate, "T:off", VCRMode.OFF, "m", store);
    assertThat(off.call("x")).isEqualTo("live");
  }

  @Test
  void promptAndMessageOverloadsDelegateAndExposeDelegate() {
    Prompt prompt = new Prompt("prompt");
    Message message = new AssistantMessage("message");
    when(delegate.call(prompt))
        .thenReturn(
            new ChatResponse(List.of(new Generation(new AssistantMessage("prompt response")))));
    when(delegate.call(any(Message[].class))).thenReturn("message response");
    VCRSpringAIChatModel off =
        new VCRSpringAIChatModel(delegate, "T:overloads", VCRMode.OFF, "m", store);

    assertThat(off.call(prompt).getResult().getOutput().getText()).isEqualTo("prompt response");
    assertThat(off.call(message)).isEqualTo("message response");
    assertThat(off.getDelegate()).isSameAs(delegate);
  }

  @Test
  void playbackRejectsWrongCassetteType() {
    store.store(
        new CassetteKey("chat", "T:wrong", 1),
        new CassetteRecord.Embedding("T:wrong", "m", 1L, new float[] {1f}));
    VCRSpringAIChatModel player =
        new VCRSpringAIChatModel(delegate, "T:wrong", VCRMode.PLAYBACK, "m", store);

    assertThatThrownBy(() -> player.call("anything"))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("Embedding");
  }

  private static ChatResponse response(String text) {
    return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
  }

  private static RateLimit rateLimit() {
    return new RateLimit() {
      @Override
      public Long getRequestsLimit() {
        return 10L;
      }

      @Override
      public Long getRequestsRemaining() {
        return 9L;
      }

      @Override
      public Duration getRequestsReset() {
        return Duration.ofSeconds(1);
      }

      @Override
      public Long getTokensLimit() {
        return 100L;
      }

      @Override
      public Long getTokensRemaining() {
        return 90L;
      }

      @Override
      public Duration getTokensReset() {
        return Duration.ofSeconds(2);
      }
    };
  }
}
