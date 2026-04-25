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

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

/** Test chat model: echoes prompt contents, counts invocations. */
final class FakeChatModel implements ChatModel {

  final AtomicInteger calls = new AtomicInteger();

  @Override
  public ChatResponse call(Prompt prompt) {
    calls.incrementAndGet();
    String reply = "echo: " + prompt.getContents();
    return new ChatResponse(List.of(new Generation(new AssistantMessage(reply))));
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    return Flux.just(call(prompt));
  }
}
