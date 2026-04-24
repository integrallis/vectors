package com.integrallis.vectors.cache.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.vectors.cache.CaffeineVectorCache;
import com.integrallis.vectors.cache.VectorCache;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.Response;
import java.util.List;
import org.junit.jupiter.api.Test;

@SuppressWarnings("removal")
class CachingChatModelTest {

  private VectorCache<String, Response<AiMessage>> newGenerateCache() {
    return CaffeineVectorCache.<String, Response<AiMessage>>builder().maximumSize(256).build();
  }

  private VectorCache<String, ChatResponse> newChatCache() {
    return CaffeineVectorCache.<String, ChatResponse>builder().maximumSize(256).build();
  }

  @Test
  void generateCachesByMessageList() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newGenerateCache(), newChatCache());

    Response<AiMessage> a = model.generate(List.of(UserMessage.from("hi")));
    Response<AiMessage> b = model.generate(List.of(UserMessage.from("hi")));
    Response<AiMessage> c = model.generate(List.of(UserMessage.from("bye")));

    assertThat(a.content().text()).contains("hi");
    assertThat(b).isSameAs(a);
    assertThat(c.content().text()).contains("bye");
    assertThat(fake.generateCalls.get()).isEqualTo(2);
  }

  @Test
  void doChatCachesByRequest() {
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newGenerateCache(), newChatCache());

    ChatRequest req = ChatRequest.builder().messages(UserMessage.from("hi")).build();
    ChatResponse first = model.doChat(req);
    ChatResponse second = model.doChat(req);

    assertThat(second).isSameAs(first);
    assertThat(fake.doChatCalls.get()).isEqualTo(1);
  }

  @Test
  void convenienceChatStringRoutesThroughGenerateCache() {
    // chat(String) is a default that bridges via doChat, which we also cache — but the default
    // chat(String) in 1.0.0-beta1 internally builds a ChatRequest, so it exercises the chatCache.
    FakeChatModel fake = new FakeChatModel();
    CachingChatModel model = new CachingChatModel(fake, newGenerateCache(), newChatCache());

    String a = model.chat("ping");
    String b = model.chat("ping");

    assertThat(a).isEqualTo(b);
    assertThat(fake.doChatCalls.get()).isEqualTo(1);
  }
}
