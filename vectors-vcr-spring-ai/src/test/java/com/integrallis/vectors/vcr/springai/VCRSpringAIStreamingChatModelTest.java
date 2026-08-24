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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.vectors.storage.backend.HeapStorageBackend;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.ExactCassetteStore;
import com.integrallis.vectors.vcr.VCRMode;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class VCRSpringAIStreamingChatModelTest {

  @Mock StreamingChatModel delegate;

  @Test
  void recordsAndReplaysAStandaloneStreamingModel() {
    CassetteStore store = new ExactCassetteStore(new HeapStorageBackend());
    Prompt prompt = new Prompt("stream it");
    when(delegate.stream(any(Prompt.class)))
        .thenReturn(Flux.just(response("one"), response("two")));

    List<ChatResponse> recorded =
        new VCRSpringAIStreamingChatModel(delegate, "T:standalone", VCRMode.RECORD, "gpt", store)
            .stream(prompt).collectList().block();
    List<ChatResponse> replayed =
        new VCRSpringAIStreamingChatModel(delegate, "T:standalone", VCRMode.PLAYBACK, "gpt", store)
            .stream(prompt).collectList().block();

    assertThat(recorded)
        .extracting(r -> r.getResult().getOutput().getText())
        .containsExactly("one", "two");
    assertThat(replayed)
        .extracting(r -> r.getResult().getOutput().getText())
        .containsExactly("one", "two");
    verify(delegate, times(1)).stream(any(Prompt.class));
  }

  private static ChatResponse response(String text) {
    return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
  }
}
