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
package com.integrallis.vectors.vcr.springai;

import com.integrallis.vectors.vcr.CassetteKey;
import com.integrallis.vectors.vcr.CassetteRecord;
import com.integrallis.vectors.vcr.CassetteStore;
import com.integrallis.vectors.vcr.VCRMode;
import com.integrallis.vectors.vcr.VCRReplayPolicy;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

/** Spring AI chat wrapper that losslessly records blocking and streaming responses. */
public final class VCRSpringAIChatModel implements ChatModel {

  private static final String TYPE_CHAT = "chat";
  private static final String TYPE_CHAT_STREAM = "chat_stream";

  private final ChatModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger callCounter = new AtomicInteger();
  private final AtomicInteger streamCounter = new AtomicInteger();

  /** Creates a VCR wrapper around a real Spring AI chat model. */
  public VCRSpringAIChatModel(
      ChatModel delegate, String testId, VCRMode mode, String modelName, CassetteStore store) {
    this.delegate = delegate;
    this.testId = testId;
    this.mode = mode;
    this.modelName = modelName;
    this.store = store;
  }

  @Override
  public ChatResponse call(Prompt prompt) {
    if (mode == VCRMode.OFF) {
      return delegate.call(prompt);
    }
    CassetteKey key = new CassetteKey(TYPE_CHAT, testId, callCounter.incrementAndGet());
    String signature =
        SpringAIRequestSignatures.chat("call", modelName, prompt, delegate.getDefaultOptions());
    Optional<CassetteRecord> existing = store.retrieve(key);
    validateType(existing, key);
    if (VCRReplayPolicy.shouldReplay(mode, existing, key, signature)) {
      CassetteRecord.Chat chat = (CassetteRecord.Chat) existing.orElseThrow();
      return SpringAIChatPayloadMapper.toResponse(chat.response());
    }
    ChatResponse response = delegate.call(prompt);
    store.store(
        key,
        new CassetteRecord.Chat(
            testId,
            modelName,
            System.currentTimeMillis(),
            prompt == null ? "" : prompt.getContents(),
            SpringAIChatPayloadMapper.toPayload(response),
            signature));
    return response;
  }

  @Override
  public String call(String message) {
    String text = message == null ? "" : message;
    return dispatchText(new Prompt(text), () -> delegate.call(text));
  }

  @Override
  public String call(Message... messages) {
    Message[] safeMessages = messages == null ? new Message[0] : messages;
    return dispatchText(new Prompt(safeMessages), () -> delegate.call(safeMessages));
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    if (mode == VCRMode.OFF) {
      return delegate.stream(prompt);
    }
    return Flux.defer(
        () -> {
          CassetteKey key =
              new CassetteKey(TYPE_CHAT_STREAM, testId, streamCounter.incrementAndGet());
          String signature =
              SpringAIRequestSignatures.chat(
                  "stream", modelName, prompt, delegate.getDefaultOptions());
          Optional<CassetteRecord> existing = store.retrieve(key);
          validateType(existing, key);
          if (VCRReplayPolicy.shouldReplay(mode, existing, key, signature)) {
            CassetteRecord.Chat chat = (CassetteRecord.Chat) existing.orElseThrow();
            return Flux.fromIterable(chat.response().streamChunks())
                .map(SpringAIChatPayloadMapper::toResponse);
          }

          List<CassetteRecord.ChatPayload> chunks = new ArrayList<>();
          return delegate.stream(prompt)
              .doOnNext(response -> chunks.add(SpringAIChatPayloadMapper.toPayload(response)))
              .doOnComplete(() -> storeStream(key, prompt, chunks, signature));
        });
  }

  @Override
  public ChatOptions getDefaultOptions() {
    return delegate.getDefaultOptions();
  }

  private void storeStream(
      CassetteKey key, Prompt prompt, List<CassetteRecord.ChatPayload> chunks, String signature) {
    CassetteRecord.ChatPayload last =
        chunks.isEmpty()
            ? new CassetteRecord.ChatPayload(
                new CassetteRecord.AiMessagePayload("", null, List.of(), null), null)
            : chunks.getLast();
    CassetteRecord.ChatPayload recorded =
        new CassetteRecord.ChatPayload(
            last.aiMessage(),
            last.metadata(),
            last.generationMetadata(),
            last.additionalGenerations(),
            List.of(),
            chunks);
    store.store(
        key,
        new CassetteRecord.Chat(
            testId,
            modelName,
            System.currentTimeMillis(),
            prompt == null ? "" : prompt.getContents(),
            recorded,
            signature));
  }

  private String dispatchText(Prompt prompt, java.util.function.Supplier<String> liveCall) {
    if (mode == VCRMode.OFF) {
      return liveCall.get();
    }
    CassetteKey key = new CassetteKey(TYPE_CHAT, testId, callCounter.incrementAndGet());
    String signature =
        SpringAIRequestSignatures.chat("call", modelName, prompt, delegate.getDefaultOptions());
    Optional<CassetteRecord> existing = store.retrieve(key);
    validateType(existing, key);
    if (VCRReplayPolicy.shouldReplay(mode, existing, key, signature)) {
      CassetteRecord.Chat chat = (CassetteRecord.Chat) existing.orElseThrow();
      return chat.response().aiMessage().text();
    }
    String response = liveCall.get();
    store.store(
        key,
        new CassetteRecord.Chat(
            testId,
            modelName,
            System.currentTimeMillis(),
            prompt.getContents(),
            new CassetteRecord.ChatPayload(
                new CassetteRecord.AiMessagePayload(response, null, List.of(), null), null),
            signature));
    return response;
  }

  private static void validateType(Optional<CassetteRecord> existing, CassetteKey key) {
    if (existing.isPresent() && !(existing.get() instanceof CassetteRecord.Chat)) {
      throw new IllegalStateException(
          "Expected Chat cassette for key "
              + key.serializedKey()
              + " but got "
              + existing.get().getClass().getSimpleName());
    }
  }

  /**
   * @return the underlying delegate for diagnostics
   */
  public ChatModel getDelegate() {
    return delegate;
  }
}
