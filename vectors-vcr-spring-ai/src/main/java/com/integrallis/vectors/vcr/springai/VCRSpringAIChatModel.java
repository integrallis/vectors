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
import com.integrallis.vectors.vcr.VCRCassetteMissingException;
import com.integrallis.vectors.vcr.VCRMode;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Spring AI {@link ChatModel} wrapper that records/replays chat responses via a {@link
 * CassetteStore}.
 */
public final class VCRSpringAIChatModel implements ChatModel {

  private static final String TYPE_CHAT = "chat";

  private final ChatModel delegate;
  private final CassetteStore store;
  private final String testId;
  private final String modelName;
  private final VCRMode mode;
  private final AtomicInteger callCounter = new AtomicInteger();

  /**
   * @param delegate real Spring AI chat model
   * @param testId test identifier
   * @param mode VCR mode
   * @param modelName model name for cassette metadata
   * @param store cassette store
   */
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
    String text =
        dispatch(
            () -> delegate.call(prompt).getResult().getOutput().getText(),
            prompt == null ? "" : String.valueOf(prompt));
    return new ChatResponse(List.of(new Generation(new AssistantMessage(text))));
  }

  @Override
  public String call(String message) {
    return dispatch(() -> delegate.call(message), message == null ? "" : message);
  }

  @Override
  public String call(Message... messages) {
    String joined = messages == null ? "" : List.of(messages).toString();
    return dispatch(() -> delegate.call(messages), joined);
  }

  private String dispatch(Supplier<String> supplier, String prompt) {
    if (mode == VCRMode.OFF) {
      return supplier.get();
    }
    CassetteKey key = new CassetteKey(TYPE_CHAT, testId, callCounter.incrementAndGet());
    if (mode.isPlaybackMode()) {
      Optional<CassetteRecord> cached = store.retrieve(key);
      if (cached.isPresent()) {
        if (cached.get() instanceof CassetteRecord.Chat c) {
          return c.response().aiMessage().text();
        }
        throw new IllegalStateException(
            "Expected Chat cassette for key "
                + key.serializedKey()
                + " but got "
                + cached.get().getClass().getSimpleName());
      }
      if (mode == VCRMode.PLAYBACK) {
        throw new VCRCassetteMissingException(key.serializedKey(), testId);
      }
    }
    String response = supplier.get();
    store.store(
        key,
        new CassetteRecord.Chat(
            testId, modelName, System.currentTimeMillis(), prompt, payload(response)));
    return response;
  }

  private static CassetteRecord.ChatPayload payload(String text) {
    return new CassetteRecord.ChatPayload(
        new CassetteRecord.AiMessagePayload(text, null, List.of(), null), null);
  }

  /**
   * @return the underlying delegate (for diagnostics)
   */
  public ChatModel getDelegate() {
    return delegate;
  }
}
