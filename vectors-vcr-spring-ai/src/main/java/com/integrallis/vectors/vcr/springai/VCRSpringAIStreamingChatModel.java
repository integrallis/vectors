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
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

/** VCR wrapper for Spring AI models that expose only {@link StreamingChatModel}. */
public final class VCRSpringAIStreamingChatModel implements StreamingChatModel {

  private static final String TYPE_CHAT_STREAM = "chat_stream";

  private final StreamingChatModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger callCounter = new AtomicInteger();

  /** Creates a VCR wrapper around a standalone Spring AI streaming model. */
  public VCRSpringAIStreamingChatModel(
      StreamingChatModel delegate,
      String testId,
      VCRMode mode,
      String modelName,
      CassetteStore store) {
    this.delegate = delegate;
    this.testId = testId;
    this.mode = mode;
    this.modelName = modelName;
    this.store = store;
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    if (mode == VCRMode.OFF) {
      return delegate.stream(prompt);
    }
    return Flux.defer(
        () -> {
          CassetteKey key =
              new CassetteKey(TYPE_CHAT_STREAM, testId, callCounter.incrementAndGet());
          String signature = SpringAIRequestSignatures.chat("stream", modelName, prompt, null);
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
              .doOnComplete(() -> store(key, prompt, signature, chunks));
        });
  }

  private void store(
      CassetteKey key, Prompt prompt, String signature, List<CassetteRecord.ChatPayload> chunks) {
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
  public StreamingChatModel getDelegate() {
    return delegate;
  }
}
