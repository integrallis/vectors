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
package com.integrallis.vectors.cache.semantic.springai;

import com.integrallis.vectors.cache.CacheFilter;
import com.integrallis.vectors.cache.SemanticCache;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.StringJoiner;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.embedding.EmbeddingModel;
import reactor.core.publisher.Flux;

/**
 * Spring AI {@link ChatModel} decorator that answers from a semantically similar earlier prompt.
 *
 * <p>The distinction from {@code CachingChatModel} matters: that one keys on a normalized prompt
 * string, so "how do I reset my password" and "password reset please" are different entries. This
 * one embeds the prompt and serves a hit when an earlier prompt is close enough, which is what a
 * semantic cache is for.
 *
 * <pre>{@code
 * ChatModel cached = new SemanticCachingChatModel(delegate, embeddingModel, semanticCache);
 * }</pre>
 *
 * <p>Two things are deliberately never served from cache.
 *
 * <p>A response carrying tool calls is an instruction to go and do something, not an answer;
 * replaying one would skip the tool and hand back a stale result. Those responses are returned but
 * not stored.
 *
 * <p>A hit produced under different request options is rejected. An embedding captures only the
 * prompt text, so without a guard a hit could return output generated at a different temperature or
 * token limit. Entries record their options as an attribute and lookups filter on it, which the
 * cache applies inside the search rather than by discarding the nearest neighbour afterwards.
 */
public class SemanticCachingChatModel implements ChatModel {

  /** Attribute recording the request options an entry was produced under. */
  public static final String OPTIONS_ATTRIBUTE = "options";

  private final ChatModel delegate;
  private final EmbeddingModel embeddingModel;
  private final SemanticCache<String> cache;

  /**
   * Wraps a chat model with a semantic cache.
   *
   * @param delegate the model to call on a miss
   * @param embeddingModel embeds prompts for similarity lookup
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
              + " generated under one set of request options from answering a request made under"
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
  public ChatResponse call(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    String contents = prompt.getContents();
    String signature = optionsSignature(prompt);
    float[] embedding = embeddingModel.embed(contents);

    String hit =
        cache
            .lookup(embedding, CacheFilter.matching(OPTIONS_ATTRIBUTE, signature))
            .map(SemanticCache.Hit::value)
            .orElse(null);
    if (hit != null) {
      return textResponse(hit);
    }

    ChatResponse response = delegate.call(prompt);
    String text = textOf(response);
    if (text != null && !carriesToolCalls(response)) {
      cache.put(contents, embedding, text, Map.of(OPTIONS_ATTRIBUTE, signature));
    }
    return response;
  }

  @Override
  public String call(String message) {
    Objects.requireNonNull(message, "message");
    return textOf(call(new Prompt(message)));
  }

  @Override
  public String call(Message... messages) {
    Objects.requireNonNull(messages, "messages");
    return textOf(call(new Prompt(List.of(messages))));
  }

  /** Streaming is not cached; a partial stream is not a complete answer. */
  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    return delegate.stream(prompt);
  }

  /**
   * Signature distinguishing requests whose options change the output.
   *
   * <p>Built from the option values rather than {@code toString()}. {@code DefaultChatOptions}
   * inherits the identity {@code toString()}, so two separately built but identical option sets
   * would otherwise produce different signatures and no request carrying options would ever hit
   * the cache.
   *
   * <p>Override to include provider-specific options a subclass knows about; the base signature
   * covers only the portable {@link ChatOptions} properties.
   *
   * @param prompt the request
   * @return a stable signature, empty when no options are set
   */
  protected String optionsSignature(Prompt prompt) {
    ChatOptions options = prompt.getOptions();
    if (options == null) {
      return "";
    }
    return new StringJoiner("|")
        .add(String.valueOf(options.getModel()))
        .add(String.valueOf(options.getTemperature()))
        .add(String.valueOf(options.getTopP()))
        .add(String.valueOf(options.getTopK()))
        .add(String.valueOf(options.getMaxTokens()))
        .add(String.valueOf(options.getFrequencyPenalty()))
        .add(String.valueOf(options.getPresencePenalty()))
        .add(String.valueOf(options.getStopSequences()))
        .toString();
  }

  private static boolean carriesToolCalls(ChatResponse response) {
    for (Generation generation : response.getResults()) {
      AssistantMessage output = generation.getOutput();
      if (output != null && output.hasToolCalls()) {
        return true;
      }
    }
    return false;
  }

  private static String textOf(ChatResponse response) {
    if (response == null || response.getResults().isEmpty()) {
      return null;
    }
    return response.getResult().getOutput().getText();
  }

  private static ChatResponse textResponse(String text) {
    return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
  }
}
