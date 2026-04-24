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
