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
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

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
}
