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
