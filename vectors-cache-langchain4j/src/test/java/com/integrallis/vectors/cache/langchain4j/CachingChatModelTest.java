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
package com.integrallis.vectors.cache.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import org.junit.jupiter.api.Test;

class CachingChatModelTest {

  private VectorCache<String, ChatResponse> newChatCache() {
    return CaffeineVectorCache.<String, ChatResponse>builder().maximumSize(256).build();
  }

  @Test
  void doChatCachesByRequest() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newChatCache());

    ChatRequest req = ChatRequest.builder().messages(UserMessage.from("hi")).build();
    ChatResponse first = model.doChat(req);
    ChatResponse second = model.doChat(req);

    assertThat(second).isSameAs(first);
    assertThat(fake.doChatCalls.get()).isEqualTo(1);
  }

  @Test
  void convenienceChatStringRoutesThroughCache() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newChatCache());

    String a = model.chat("ping");
    String b = model.chat("ping");

    assertThat(a).isEqualTo(b);
    assertThat(fake.doChatCalls.get()).isEqualTo(1);
  }
}
