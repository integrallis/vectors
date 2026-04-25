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

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.Response;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/** Test chat model: echoes the messages, counts separate invocations per path. */
@SuppressWarnings("removal")
final class FakeChatModel implements ChatLanguageModel {

  final AtomicInteger generateCalls = new AtomicInteger();
  final AtomicInteger doChatCalls = new AtomicInteger();

  @Override
  public Response<AiMessage> generate(List<ChatMessage> messages) {
    generateCalls.incrementAndGet();
    return Response.from(AiMessage.from("echo: " + messages));
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    doChatCalls.incrementAndGet();
    return ChatResponse.builder().aiMessage(AiMessage.from("echo: " + request.messages())).build();
  }
}
