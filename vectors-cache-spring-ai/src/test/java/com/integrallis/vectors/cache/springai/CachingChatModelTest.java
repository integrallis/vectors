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
package com.integrallis.vectors.cache.springai;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

class CachingChatModelTest {

  private VectorCache<String, ChatResponse> newCache() {
    return CaffeineVectorCache.<String, ChatResponse>builder().maximumSize(256).build();
  }

  @Test
  void promptCallCachesByContents() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newCache());

    ChatResponse first = model.call(new Prompt("hello"));
    ChatResponse second = model.call(new Prompt("hello"));

    assertThat(first.getResult().getOutput().getText()).isEqualTo("echo: hello");
    assertThat(second).isSameAs(first);
    assertThat(fake.calls.get()).isEqualTo(1);
  }

  @Test
  void differentContentProducesDifferentCacheEntries() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newCache());

    model.call(new Prompt("hello"));
    model.call(new Prompt("world"));
    model.call(new Prompt("hello"));

    assertThat(fake.calls.get()).isEqualTo(2);
  }

  @Test
  void stringCallReturnsCachedText() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newCache());

    String a = model.call("ping");
    String b = model.call("ping");

    assertThat(a).isEqualTo("echo: ping");
    assertThat(b).isEqualTo("echo: ping");
    assertThat(fake.calls.get()).isEqualTo(1);
  }

  @Test
  void messagesCallCaches() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newCache());

    String a = model.call(new UserMessage("q"));
    String b = model.call(new UserMessage("q"));

    assertThat(a).startsWith("echo:");
    assertThat(b).isEqualTo(a);
    assertThat(fake.calls.get()).isEqualTo(1);
  }

  @Test
  void streamIsNotCached() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newCache());

    model.stream(new Prompt("x")).blockFirst();
    model.stream(new Prompt("x")).blockFirst();

    assertThat(fake.calls.get()).isEqualTo(2);
  }
}
