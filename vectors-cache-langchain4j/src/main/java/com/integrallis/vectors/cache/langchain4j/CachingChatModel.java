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

import com.integrallis.vectors.cache.VectorCache;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.Response;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Caching decorator around a LangChain4j {@link ChatLanguageModel}. The mandatory {@link
 * #generate(List)} entry point (used internally by the legacy {@code chat(...)} bridge methods) and
 * the newer {@link #doChat(ChatRequest)} path both consult the supplied {@link VectorCache}.
 *
 * <p>Cache keys are derived from the message list (and, for {@code doChat}, from the full {@link
 * ChatRequest}) via the supplied extractors. The default extractors use {@code toString()}; supply
 * SHA-256 wrappers for workloads where prompt identity must be bit-stable.
 */
@SuppressWarnings("removal")
public class CachingChatModel implements ChatLanguageModel {

  private final ChatLanguageModel delegate;
  private final VectorCache<String, Response<AiMessage>> generateCache;
  private final VectorCache<String, ChatResponse> chatCache;
  private final Function<List<ChatMessage>, String> messagesKey;
  private final Function<ChatRequest, String> requestKey;
  private final UnaryOperator<String> keyFn;

  /** Convenience constructor using default key extractors. */
  public CachingChatModel(
      ChatLanguageModel delegate,
      VectorCache<String, Response<AiMessage>> generateCache,
      VectorCache<String, ChatResponse> chatCache) {
    this(
        delegate,
        generateCache,
        chatCache,
        CachingChatModel::defaultMessagesKey,
        CachingChatModel::defaultRequestKey,
        UnaryOperator.identity());
  }

  /**
   * @param delegate non-null underlying chat model
   * @param generateCache cache for the legacy {@link #generate(List)} path
   * @param chatCache cache for the {@link #doChat(ChatRequest)} path
   * @param messagesKey extractor for the legacy path
   * @param requestKey extractor for the ChatRequest path
   * @param keyFn post-normalization of every key (e.g. hashing)
   */
  public CachingChatModel(
      ChatLanguageModel delegate,
      VectorCache<String, Response<AiMessage>> generateCache,
      VectorCache<String, ChatResponse> chatCache,
      Function<List<ChatMessage>, String> messagesKey,
      Function<ChatRequest, String> requestKey,
      UnaryOperator<String> keyFn) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.generateCache = Objects.requireNonNull(generateCache, "generateCache");
    this.chatCache = Objects.requireNonNull(chatCache, "chatCache");
    this.messagesKey = Objects.requireNonNull(messagesKey, "messagesKey");
    this.requestKey = Objects.requireNonNull(requestKey, "requestKey");
    this.keyFn = Objects.requireNonNull(keyFn, "keyFn");
  }

  /**
   * @return the wrapped delegate
   */
  public final ChatLanguageModel delegate() {
    return delegate;
  }

  @Override
  public Response<AiMessage> generate(List<ChatMessage> messages) {
    Objects.requireNonNull(messages, "messages");
    String key = keyFn.apply(messagesKey.apply(messages));
    return generateCache.getOrCompute(key, k -> delegate.generate(messages));
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    String key = keyFn.apply(requestKey.apply(request));
    return chatCache.getOrCompute(key, k -> delegate.doChat(request));
  }

  /** Default messages-key extractor: the {@code toString()} of the message list. */
  public static String defaultMessagesKey(List<ChatMessage> messages) {
    return String.valueOf(messages);
  }

  /**
   * Default {@link ChatRequest} key extractor: the messages' {@code toString()} joined with the
   * request parameters' {@code toString()} under a NUL separator.
   */
  public static String defaultRequestKey(ChatRequest request) {
    return String.valueOf(request.messages()) + "\u0000" + String.valueOf(request.parameters());
  }
}
