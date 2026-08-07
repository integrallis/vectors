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
package com.integrallis.vectors.cache.semantic.langchain4j;

import com.integrallis.vectors.cache.CacheFilter;
import com.integrallis.vectors.cache.SemanticCache;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * LangChain4j {@link ChatModel} decorator that answers from a semantically similar earlier request.
 *
 * <p>The distinction from {@code CachingChatModel} matters: that one keys on the messages' {@code
 * toString()}, so "how do I reset my password" and "I forgot my password" are separate entries.
 * This one embeds the request and serves a hit when an earlier request is close enough, which is
 * what a semantic cache is for.
 *
 * <pre>{@code
 * ChatModel cached = new SemanticCachingChatModel(delegate, embeddingModel, semanticCache);
 * }</pre>
 *
 * <p>A response carrying tool execution requests is never cached: it asks the caller to go and do
 * something, and replaying it would skip the tool and return a stale result.
 *
 * <p>A hit produced under different request parameters is rejected. An embedding captures only the
 * message text, so without a guard a hit could return output generated at a different temperature
 * or token limit. Entries record their parameters as an attribute and lookups filter on it, which
 * the cache applies inside the search rather than by discarding the nearest neighbour afterwards.
 */
public class SemanticCachingChatModel implements ChatModel {

  /** Attribute recording the request parameters an entry was produced under. */
  public static final String PARAMETERS_ATTRIBUTE = "parameters";

  private final ChatModel delegate;
  private final EmbeddingModel embeddingModel;
  private final SemanticCache<String> cache;

  /**
   * Wraps a chat model with a semantic cache.
   *
   * @param delegate the model to call on a miss
   * @param embeddingModel embeds requests for similarity lookup
   * @param cache similarity cache holding previous completions; must support entry attributes
   * @throws IllegalArgumentException if the cache cannot store attributes
   */
  public SemanticCachingChatModel(
      ChatModel delegate, EmbeddingModel embeddingModel, SemanticCache<String> cache) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.embeddingModel = Objects.requireNonNull(embeddingModel, "embeddingModel");
    this.cache = Objects.requireNonNull(cache, "cache");
    if (!cache.supportsAttributes()) {
      throw new IllegalArgumentException(
          cache.getClass().getName()
              + " does not store entry attributes, which this decorator needs to keep a completion"
              + " generated under one set of request parameters from answering a request made under"
              + " another");
    }
  }

  /**
   * Returns the wrapped model.
   *
   * @return the delegate
   */
  public final ChatModel delegate() {
    return delegate;
  }

  /**
   * Returns the backing cache, for stats and invalidation.
   *
   * @return the semantic cache
   */
  public final SemanticCache<String> cache() {
    return cache;
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    Objects.requireNonNull(request, "request");
    String text = textOf(request.messages());
    String signature = parameterSignature(request);
    float[] embedding = embeddingModel.embed(text).content().vector();

    String hit =
        cache
            .lookup(embedding, CacheFilter.matching(PARAMETERS_ATTRIBUTE, signature))
            .map(SemanticCache.Hit::value)
            .orElse(null);
    if (hit != null) {
      return ChatResponse.builder().aiMessage(AiMessage.from(hit)).build();
    }

    ChatResponse response = delegate.doChat(request);
    AiMessage message = response.aiMessage();
    if (message != null && !message.hasToolExecutionRequests() && message.text() != null) {
      cache.put(text, embedding, message.text(), Map.of(PARAMETERS_ATTRIBUTE, signature));
    }
    return response;
  }

  /**
   * Signature distinguishing requests whose parameters change the output.
   *
   * @param request the request
   * @return a stable signature
   */
  protected String parameterSignature(ChatRequest request) {
    return String.valueOf(request.parameters());
  }

  /**
   * Text embedded for similarity lookup.
   *
   * <p>Uses the whole conversation, not only the final turn: a system prompt or an earlier answer
   * changes what a follow-up question means, and a cache that ignored them would serve confidently
   * wrong replies.
   *
   * @param messages the conversation
   * @return concatenated message text
   */
  protected String textOf(List<ChatMessage> messages) {
    StringBuilder text = new StringBuilder();
    for (ChatMessage message : messages) {
      if (!text.isEmpty()) {
        text.append('\n');
      }
      if (message instanceof UserMessage user) {
        text.append(user.singleText());
      } else if (message instanceof AiMessage ai && ai.text() != null) {
        text.append(ai.text());
      } else {
        text.append(message);
      }
    }
    return text.toString();
  }

}
