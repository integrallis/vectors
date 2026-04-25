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

import com.integrallis.vectors.cache.VectorCache;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

/**
 * Caching decorator around a Spring AI {@link ChatModel}. {@link #call(Prompt)}, {@link
 * #call(String)}, and {@link #call(Message...)} are cached by a normalized key derived from the
 * call arguments. {@link #stream(Prompt)} is not cached — streaming responses are forwarded
 * verbatim to the delegate to preserve backpressure.
 *
 * <p>The key functions are pluggable. The default {@code promptKey} is the result of {@code
 * prompt.getContents()} plus the {@code toString()} of the prompt options; callers needing stronger
 * isolation (e.g. SHA-256 of the full prompt) can pass a custom function.
 *
 * <p>On a cache hit the stored {@link ChatResponse} is returned by reference. Downstream consumers
 * should treat {@link ChatResponse} as effectively immutable — the Spring AI API exposes only read
 * accessors on it, so sharing across calls is safe in practice.
 */
public class CachingChatModel implements ChatModel {

  private final ChatModel delegate;
  private final VectorCache<String, ChatResponse> cache;
  private final Function<Prompt, String> promptKey;
  private final UnaryOperator<String> keyFn;

  /** Convenience constructor using the default prompt key extractor. */
  public CachingChatModel(ChatModel delegate, VectorCache<String, ChatResponse> cache) {
    this(delegate, cache, CachingChatModel::defaultPromptKey, UnaryOperator.identity());
  }

  /**
   * @param delegate non-null underlying chat model
   * @param cache non-null response cache
   * @param promptKey function extracting a cache key from a {@link Prompt}
   * @param keyFn additional normalization applied to every computed key (e.g. hashing)
   */
  public CachingChatModel(
      ChatModel delegate,
      VectorCache<String, ChatResponse> cache,
      Function<Prompt, String> promptKey,
      UnaryOperator<String> keyFn) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.cache = Objects.requireNonNull(cache, "cache");
    this.promptKey = Objects.requireNonNull(promptKey, "promptKey");
    this.keyFn = Objects.requireNonNull(keyFn, "keyFn");
  }

  /**
   * @return the wrapped delegate
   */
  public final ChatModel delegate() {
    return delegate;
  }

  @Override
  public ChatResponse call(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    String key = keyFn.apply(promptKey.apply(prompt));
    return cache.getOrCompute(key, k -> delegate.call(prompt));
  }

  @Override
  public String call(String message) {
    Objects.requireNonNull(message, "message");
    String key = keyFn.apply("str:" + message);
    ChatResponse response = cache.getOrCompute(key, k -> delegate.call(new Prompt(message)));
    return response.getResult().getOutput().getText();
  }

  @Override
  public String call(Message... messages) {
    Objects.requireNonNull(messages, "messages");
    String key = keyFn.apply("msgs:" + List.of(messages).toString());
    ChatResponse response =
        cache.getOrCompute(key, k -> delegate.call(new Prompt(List.of(messages))));
    return response.getResult().getOutput().getText();
  }

  /** Streaming is not cached; delegates verbatim. */
  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    return delegate.stream(prompt);
  }

  /**
   * Default prompt-key extractor: concatenates {@code prompt.getContents()} with the {@code
   * toString()} of any configured chat options.
   */
  public static String defaultPromptKey(Prompt prompt) {
    String contents = prompt.getContents();
    Object options = prompt.getOptions();
    return options == null ? contents : contents + "\u0000" + options;
  }
}
