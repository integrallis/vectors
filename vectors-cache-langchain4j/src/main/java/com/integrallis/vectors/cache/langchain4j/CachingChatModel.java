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
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Caching decorator around a LangChain4j {@link ChatModel}. The {@link #doChat(ChatRequest)} entry
 * point (used internally by the convenience {@code chat(...)} methods) consults the supplied {@link
 * VectorCache}.
 *
 * <p>Cache keys are derived from the {@link ChatRequest} via the supplied extractor. The default
 * extractor uses {@code toString()}; supply a SHA-256 wrapper for workloads where prompt identity
 * must be bit-stable.
 */
public class CachingChatModel implements ChatModel {

  private final ChatModel delegate;
  private final VectorCache<String, ChatResponse> chatCache;
  private final Function<ChatRequest, String> requestKey;
  private final UnaryOperator<String> keyFn;

  /** Convenience constructor using default key extractors. */
  public CachingChatModel(ChatModel delegate, VectorCache<String, ChatResponse> chatCache) {
    this(delegate, chatCache, CachingChatModel::defaultRequestKey, UnaryOperator.identity());
  }

  /**
   * @param delegate non-null underlying chat model
   * @param chatCache cache for the {@link #doChat(ChatRequest)} path
   * @param requestKey extractor for the ChatRequest path
   * @param keyFn post-normalization of every key (e.g. hashing)
   */
  public CachingChatModel(
      ChatModel delegate,
      VectorCache<String, ChatResponse> chatCache,
      Function<ChatRequest, String> requestKey,
      UnaryOperator<String> keyFn) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.chatCache = Objects.requireNonNull(chatCache, "chatCache");
    this.requestKey = Objects.requireNonNull(requestKey, "requestKey");
    this.keyFn = Objects.requireNonNull(keyFn, "keyFn");
  }

  /**
   * @return the wrapped delegate
   */
  public final ChatModel delegate() {
    return delegate;
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    String key = keyFn.apply(requestKey.apply(request));
    return chatCache.getOrCompute(key, k -> delegate.doChat(request));
  }

  /**
   * Default {@link ChatRequest} key extractor: the messages' {@code toString()} joined with the
   * request parameters' {@code toString()} under a NUL separator.
   */
  public static String defaultRequestKey(ChatRequest request) {
    return String.valueOf(request.messages()) + "\u0000" + String.valueOf(request.parameters());
  }
}
